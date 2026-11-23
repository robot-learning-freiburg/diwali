

import logging
import math
import os
import pickle

import einops
import imageio
import numpy as np
import torch

import wandb

log = logging.getLogger(__name__)
from diwali.agent.finetune.train_ppo_agent import TrainPPOAgent
from diwali.utils.scheduler import CosineAnnealingWarmupRestarts
from diwali.utils.timer import Timer

ACTION_CHUNK_SIZE = 5


class TrainMBPPODiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )

        assert self.logprob_batch_size % self.n_envs == 0, "logprob_batch_size must be divisible by n_envs"

    def get_init_obs(self, n_envs):
        """
        Get initial observations for the environment.
        This function is used to reset a dummy environment and get the initial observations.
        """
        if self.env_type == "calvin":
            obs = self.get_init_obs_calvin(n_envs)
        elif self.env_type == "real":
            obs = self.get_init_obs_real(n_envs)
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")
        return obs
    
    def get_init_obs_real(self, n_envs):
        """Sample RGB start frames from the real-world dataset (no physical env)."""
        from diwali.env.runner.visionwm_real import _as_chw

        if self.real_starts is None:
            raise ValueError("Real-world start frames were not loaded.")
        n_starts = len(self.real_starts["robot_obs"])
        inds = np.random.randint(0, n_starts, size=n_envs)
        robot_obs, rgb_static, rgb_gripper = [], [], []
        for idx in inds:
            robot = np.asarray(self.real_starts["robot_obs"][idx], dtype=np.float32)
            if robot.ndim > 1:
                robot = robot[-1]
            robot = robot.reshape(-1)
            robot_obs.append(robot[:7] if robot.shape[0] >= 7 else robot)
            rgb_s = np.asarray(self.real_starts["rgb_static"][idx])
            rgb_g = np.asarray(self.real_starts["rgb_gripper"][idx])
            if rgb_s.ndim == 4:
                rgb_s = rgb_s[-1]
                rgb_g = rgb_g[-1]
            rgb_static.append(_as_chw(rgb_s))
            rgb_gripper.append(_as_chw(rgb_g))
        return {
            "robot_obs": np.stack(robot_obs),
            "rgb_static": np.stack(rgb_static),
            "rgb_gripper": np.stack(rgb_gripper),
        }

    def get_init_obs_calvin(self, n_envs):
        obs = [self.venv.reset()[0] for _ in range(n_envs)]
        robot_obs = np.array([ob["robot_obs"][-1] for ob in obs])
        rgb_static = np.array([ob["rgb_static"][-1] for ob in obs])
        rgb_gripper = np.array([ob["rgb_gripper"][-1] for ob in obs])

        obs = {
            "robot_obs": robot_obs,
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper,
        }
        return obs

    def get_terminated_from_reward(self, reward):
        terminated = np.zeros_like(reward)
        terminated[reward > 0] = 1
        return terminated.astype(bool)

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_wm_train_step = 0
        self.video_writer = None
        self.itr = 0

        while self.itr <= self.n_train_itr:
            # Define train or eval - all envs restart
            if not self.eval_in_sim:
                eval_mode = False
            elif self.itr == 0 or self.itr == self.n_train_itr:
                eval_mode = True
            else:
                eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            if self.itr == 0 and not self.eval_in_sim:
                log.info("Simulator eval disabled (train.eval_in_sim=false); training only.")
            self.model.eval() if eval_mode else self.model.train()

            if not eval_mode:
                done_venv = np.zeros((1, self.n_envs))
                obs_trajs = {
                    "state": np.zeros(
                        (
                            self.n_steps,
                            self.n_envs,
                            self.n_cond_step,
                            self.obs_dim,
                        )
                    )
                }
                chains_trajs = np.zeros(
                    (
                        self.n_steps,
                        self.n_envs,
                        self.model.ft_denoising_steps + 1,
                        self.horizon_steps,
                        self.action_dim,
                    )
                )
                terminated_trajs = np.zeros((self.n_steps, self.n_envs))
                reward_trajs = np.zeros((self.n_steps, self.n_envs))
                firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
                rgb_static_trajs = np.zeros((self.n_steps, self.n_envs, 64, 64, 3))
                rgb_gripper_trajs = np.zeros((self.n_steps, self.n_envs, 64, 64, 3))

                # Initialize environment
                prev_obs_venv = self.get_init_obs(self.n_envs)
                wm_step_counter = np.zeros((self.n_envs)).astype(int)
                firsts_trajs[0] = 1

                # Per-env count of CONSECUTIVE positive-reward steps, for windowed
                # success-termination (see terminate_success_window below).
                consec_success = np.zeros((self.n_envs)).astype(int)
                _term_window = int(self.cfg.train.get("terminate_success_window", 5))

                # Option C: confirm success over the window but pay the reward
                # ONCE (on the confirmation frame) instead of per success frame.
                _reward_on_confirm = bool(self.cfg.train.get("reward_on_confirmation_only", True))

                if self.wme is not None:
                    for key in prev_obs_venv:
                        prev_obs_venv[key] = np.expand_dims(prev_obs_venv[key], 1)

                    wm_features, out_state = self.wme.get_zero_wm_features(prev_obs_venv, self.device)
                    prev_obs_venv = wm_features.cpu().numpy()
                    in_state = out_state

                if self.save_full_observations:  # state-only
                    obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                    obs_full_trajs = np.vstack((obs_full_trajs, prev_obs_venv[:, -1][None]))

                # --- action-smoothness OBSERVATION (logging-only; no penalty yet) ---
                # rel_actions ~ per-step velocity. We track, on the continuous dims
                # [:6] (the gripper dim is excluded), within each 4-action chunk:
                #   |a|        velocity magnitude
                #   |Δa|       acceleration (consecutive action difference)
                #   Σ|Δa|^2    the would-be jerk-penalty term (per chunk)
                # so the jerk scale can be observed before choosing a coefficient.
                _act_vel_sum = _act_acc_sum = _act_jerkpen_sum = 0.0
                _act_vel_max = _act_acc_max = 0.0
                _act_vel_cnt = _act_acc_cnt = _act_chunk_cnt = 0
                # inter-chunk boundary jerk: |a1[s] - a4[s-1]|, EXCLUDING envs that
                # reset between the two chunks (cross-episode seam is not real jerk).
                _act_bacc_sum = _act_bjerkpen_sum = _act_bacc_max = 0.0
                _act_bacc_cnt = 0
                _prev_last_action = None   # a4[:6] of the previous chunk
                _prev_done = None          # done_venv of the previous step (reset mask)
                # Episode-MEAN action-jerk penalty (length-independent): accumulate
                # each env's per-chunk jerk Σ|Δa|^2 over its episode, then subtract
                # coef * mean_jerk ONCE at termination. Because it uses the MEAN (not
                # the per-step sum), it does NOT scale with episode length and so adds
                # no living-cost / shortening incentive. 0 = off. train.action_jerk_coef.
                _act_jerk_coef = float(self.cfg.train.get("action_jerk_coef", 0.0) or 0.0)
                _ep_jerk_sum = np.zeros(self.n_envs)   # running Σ jerk per env (this episode)
                _ep_jerk_cnt = np.zeros(self.n_envs)   # running chunk count per env
                _term_pen_sum = 0.0; _term_pen_cnt = 0  # logging: mean applied terminal penalty

                for step in range(self.n_steps):
                    if step % 10 == 0:
                        print(f"Processed env step {step} of {self.n_steps}")

                    with torch.no_grad():
                        cond = {"state": torch.from_numpy(prev_obs_venv).float().to(self.device)}
                        samples = self.model(
                            cond=cond,
                            deterministic=eval_mode,
                            return_chain=True,
                        )
                        output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
                        chains_venv = samples.chains.cpu().numpy()  # n_env x denoising x horizon x act
                    action_venv = output_venv[:, : self.act_steps]
                    # action smoothness: per-env jerk Σ|Δa|^2 (within-chunk + reset-
                    # masked boundary) used for the penalty, plus the logging sums.
                    # Continuous dims [:6] only (gripper excluded).
                    _a = action_venv[..., :6]                          # (n_env, act_steps, 6)
                    _vel = np.linalg.norm(_a, axis=-1)                 # |a| per action
                    _act_vel_sum += float(_vel.sum()); _act_vel_cnt += int(_vel.size)
                    _act_vel_max = max(_act_vel_max, float(_vel.max()))
                    _jerk_per_env = np.zeros(_a.shape[0])              # Σ|Δa|^2 per env (within + boundary)
                    if _a.shape[1] > 1:
                        _da = np.diff(_a, axis=1)                      # ~ acceleration (within-chunk)
                        _acc = np.linalg.norm(_da, axis=-1)            # |Δa|
                        _act_acc_sum += float(_acc.sum()); _act_acc_cnt += int(_acc.size)
                        _act_acc_max = max(_act_acc_max, float(_acc.max()))
                        _jw = (_da ** 2).sum(axis=(1, 2))             # (n_env,) within-chunk Σ|Δa|^2
                        _act_jerkpen_sum += float(_jw.sum()); _act_chunk_cnt += int(_a.shape[0])
                        _jerk_per_env = _jerk_per_env + _jw
                    if _prev_last_action is not None:                  # inter-chunk seam a1[s]-a4[s-1]
                        _cont = (~_prev_done.astype(bool)) if _prev_done is not None \
                            else np.ones(_a.shape[0], dtype=bool)
                        _bd = _a[:, 0, :] - _prev_last_action          # (n_env, 6)
                        _bpen = (_bd ** 2).sum(axis=1)                 # (n_env,)
                        _jerk_per_env = _jerk_per_env + np.where(_cont, _bpen, 0.0)  # masked at resets
                        if _cont.any():
                            _bacc = np.linalg.norm(_bd, axis=-1)[_cont]
                            _act_bacc_sum += float(_bacc.sum()); _act_bacc_cnt += int(_bacc.size)
                            _act_bacc_max = max(_act_bacc_max, float(_bacc.max()))
                            _act_bjerkpen_sum += float(_bpen[_cont].sum())
                    # accumulate this chunk's jerk into the per-episode running sum
                    _ep_jerk_sum = _ep_jerk_sum + _jerk_per_env
                    _ep_jerk_cnt = _ep_jerk_cnt + 1.0
                    # Apply multi-step in wm action
                    (
                        obs_venv,
                        reward_venv,
                        dcd_rgb_static,
                        dcd_rgb_gripper,
                    ) = self.wme.multi_step(
                        torch.from_numpy(prev_obs_venv).squeeze().float().to(self.device),
                        torch.from_numpy(action_venv).reshape(self.n_envs, -1).float().to(self.device),
                    )
                    obs_venv = obs_venv.cpu().numpy()
                    wm_step_counter += self.act_steps * ACTION_CHUNK_SIZE
                    truncated_venv = wm_step_counter >= self.max_episode_steps
                    # terminated_venv = self.get_terminated_from_reward(reward_venv)


                    # Windowed success-termination: only terminate once reward has
                    # been positive for ``terminate_success_window`` CONSECUTIVE
                    # steps (default 1 == terminate on the first success). A larger
                    # window keeps the imagined episode running through transient
                    # single-step successes, stopping only on sustained success
                    # (and otherwise running to truncation). The intermediate
                    # success steps stay non-terminal, so GAE bootstraps through
                    # them and the policy is rewarded for *holding* success.
                    _succ_now = self.get_terminated_from_reward(reward_venv)
                    consec_success = np.where(_succ_now, consec_success + 1, 0)
                    terminated_venv = consec_success >= _term_window
                    if _reward_on_confirm:
                        # Confirm-then-reward-ONCE: pay the success reward only on
                        # the confirmation frame (the window-th consecutive
                        # success) and zero the pre-confirmation success frames,
                        # so the policy is rewarded for ACHIEVING sustained
                        # success, not for camping in it. terminated_venv is the
                        # per-step confirmation flag.
                        reward_venv = np.where(terminated_venv, reward_venv, 0.0)
                    done_venv = terminated_venv | truncated_venv

                    # Episode-MEAN jerk penalty, applied ONCE at termination: subtract
                    # coef * (mean Σ|Δa|^2 over the just-finished episode). Length-
                    # INDEPENDENT (mean, not sum) -> no living-cost / shortening
                    # incentive. Applied AFTER success detection so it never suppresses
                    # the success signal. Accumulators are reset for the ended envs.
                    if np.any(done_venv):
                        _dm = done_venv.astype(bool)
                        if _act_jerk_coef > 0.0:
                            _mean_jerk = _ep_jerk_sum[_dm] / np.maximum(_ep_jerk_cnt[_dm], 1.0)
                            _term_pen = _act_jerk_coef * _mean_jerk
                            reward_venv[_dm] = reward_venv[_dm] - _term_pen
                            _term_pen_sum += float(_term_pen.sum()); _term_pen_cnt += int(_dm.sum())
                        _ep_jerk_sum[_dm] = 0.0
                        _ep_jerk_cnt[_dm] = 0.0

                    obs_trajs["state"][step] = prev_obs_venv
                    chains_trajs[step] = chains_venv
                    reward_trajs[step] = reward_venv
                    terminated_trajs[step] = terminated_venv
                    firsts_trajs[step + 1] = done_venv

                    # To render videos later
                    rgb_static_trajs[step] = dcd_rgb_static
                    rgb_gripper_trajs[step] = dcd_rgb_gripper

                    cnt_wm_train_step += self.n_envs * self.act_steps

                    # If an episode is done, reset environment for that env
                    if sum(done_venv) > 0:
                        new_obs_venv = self.get_init_obs(int(sum(done_venv)))

                        # WM Encoder
                        if self.wme is not None:
                            for key in new_obs_venv:
                                new_obs_venv[key] = np.expand_dims(new_obs_venv[key], 1)
                            wm_features, out_state = self.wme.get_zero_wm_features(new_obs_venv, self.device)
                            new_obs_venv = wm_features.squeeze().cpu().numpy()

                        obs_venv[done_venv.astype(bool)] = new_obs_venv
                        wm_step_counter[done_venv.astype(bool)] = 0
                        consec_success[done_venv.astype(bool)] = 0

                    prev_obs_venv = np.expand_dims(obs_venv, 1)

                episodes_start_end = []
                for env_ind in range(self.n_envs):
                    env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                    for i in range(len(env_steps) - 1):
                        start = env_steps[i]
                        end = env_steps[i + 1]
                        if end - start > 1:
                            episodes_start_end.append((env_ind, start, end - 1))
                if len(episodes_start_end) > 0:
                    reward_trajs_split = [
                        reward_trajs[start : end + 1, env_ind]
                        for env_ind, start, end in episodes_start_end
                    ]
                    num_episode_finished = len(reward_trajs_split)
                    episode_reward = np.array(
                        [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                    )
                    if (
                        self.furniture_sparse_reward
                    ):  # only for furniture tasks, where reward only occurs in one env step
                        episode_best_reward = episode_reward
                    else:
                        episode_best_reward = np.array(
                            [
                                np.max(reward_traj) / self.act_steps
                                for reward_traj in reward_trajs_split
                            ]
                        )
                    avg_episode_reward = np.sum(reward_trajs) / np.sum(firsts_trajs[1:, :])
                    avg_best_reward = np.mean(episode_best_reward)
                    success_rate = np.mean(
                        episode_best_reward >= self.best_reward_threshold_for_success
                    )
                    avg_episode_length = np.mean(np.array([end - start for _, start, end in episodes_start_end]))
                    obs_venv = {"state": obs_venv}
                
                # action-smoothness summaries (logging-only): mean |a| (velocity),
                # mean |Δa| (acceleration), mean Σ|Δa|^2 per chunk (would-be penalty).
                avg_action_vel = _act_vel_sum / max(_act_vel_cnt, 1)
                avg_action_acc = _act_acc_sum / max(_act_acc_cnt, 1)
                avg_action_jerk_pen = _act_jerkpen_sum / max(_act_chunk_cnt, 1)
                avg_action_bacc = _act_bacc_sum / max(_act_bacc_cnt, 1)          # mean |a1[s]-a4[s-1]|
                avg_action_bjerk_pen = _act_bjerkpen_sum / max(_act_bacc_cnt, 1)  # mean |Δ|^2 per seam
                avg_term_jerk_pen = _term_pen_sum / max(_term_pen_cnt, 1)        # mean APPLIED terminal penalty / episode

                # Randomly choose an episode and render it
                if self.itr % self.render_freq == 0:
                    train_rgb_static_path = os.path.join(self.render_dir, f"train-rgb-static-{self.itr}.mp4")
                    train_rgb_gripper_path = os.path.join(self.render_dir, f"train-rgb-gripper-{self.itr}.mp4")

                    if self.video_writer is None:
                        self.video_writer_rgb_static = imageio.get_writer(
                            train_rgb_static_path, fps=int(30 / self.act_steps)
                        )
                        self.video_writer_rgb_gripper = imageio.get_writer(
                            train_rgb_gripper_path, fps=int(30 / self.act_steps)
                        )
                    train_env_id = np.random.randint(0, self.n_envs)
                    ep_ends = np.where(firsts_trajs[1:, train_env_id])[0]
                    c_ = 0
                    failed = False
                    while len(ep_ends) == 0:
                        train_env_id = np.random.randint(0, self.n_envs)
                        ep_ends = np.where(firsts_trajs[1:, train_env_id])[0]
                        c_ += 1
                        if c_ > self.n_envs:
                            log.info("No terminated episode found for rendering")
                            failed = True
                            break
                    if not failed and len(ep_ends) > 1:
                        rand_ind = np.random.randint(1, len(ep_ends))
                        start_ind = ep_ends[rand_ind - 1] + 1
                        end_ind = ep_ends[rand_ind] + 1
                        for idx in range(start_ind, end_ind):
                            self.video_writer_rgb_static.append_data(
                                rgb_static_trajs[idx, train_env_id].astype("uint8")
                            )
                            self.video_writer_rgb_gripper.append_data(
                                rgb_gripper_trajs[idx, train_env_id].astype("uint8")
                            )
                        self.video_writer_rgb_static.close()
                        self.video_writer_rgb_gripper.close()
                        self.video_writer_rgb_static = None
                        self.video_writer_rgb_gripper = None
                    else:
                        self.video_writer_rgb_static = None
                        self.video_writer_rgb_gripper = None

            else:
                episode_rewards = []
                episode_lengths = []
                options_venv = None
                rand_ind = None
                total_episodes_eval = 0
                if self.env_type == "real":
                    if getattr(self, "env_runner", None) is None:
                        log.warning("eval_in_sim is set for real but no env_runner; skipping simulator eval.")
                    else:
                        self.env_runner.run(
                            self.itr,
                            self.model,
                            self.env,
                            self.wme,
                            getattr(self, "action_decoder", None),
                        )
                    avg_episode_reward = np.nan
                    avg_episode_length = np.nan
                    success_rate = np.nan
                    num_episode_finished = 0
                    avg_best_reward = np.nan
                elif self.env_type == "calvin":
                    robot_obs = self.env.robot_obs
                    scene_obs = self.env.scene_obs
                    total_episodes_eval = robot_obs.shape[0]
                    options_venv = [{} for _ in range(total_episodes_eval)]
                    if self.n_render == 1:
                        rand_ind = np.random.randint(0, total_episodes_eval)
                        options_venv[rand_ind] = {
                            "video_path": os.path.join(
                                self.render_dir,
                                f"itr-{self.itr}_trial-{rand_ind}.mp4",
                            )
                        }
                    elif self.n_render == -1:
                        for i in range(total_episodes_eval):
                            options_venv[i] = {
                                "video_path": os.path.join(
                                    self.render_dir,
                                    f"itr-{self.itr}_trial-{i}.mp4",
                                )
                            }

                    for i in range(total_episodes_eval):
                        if i % 10 == 0:
                            print(f"Processed episode {i} of {total_episodes_eval}")
                        prev_obs_venv = {}
                        prev_obs_venv["state"], _ = self.env.reset(
                            robot_obs=robot_obs[i],
                            scene_obs=scene_obs[i],
                            options=options_venv[i],
                        )

                        # WM Encoder
                        if self.wme is not None:
                            for key in prev_obs_venv["state"]:
                                prev_obs_venv["state"][key] = np.expand_dims(
                                    np.expand_dims(prev_obs_venv["state"][key][-1, :], 0), 0
                                )

                            wm_features, out_state = self.wme.get_zero_wm_features(prev_obs_venv["state"], self.device)
                            prev_obs_venv["state"] = wm_features.cpu().numpy()
                            in_state = out_state
                            prev_done_venv = np.zeros((1)).astype(bool)

                        for step in range(self.n_steps):
                            # Select action
                            with torch.no_grad():
                                cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                                samples = self.model(
                                    cond=cond,
                                    deterministic=eval_mode,
                                    return_chain=True,
                                )
                            if self.action_decoder is not None:
                                lat_act = samples.trajectories[:, 0, :]  # (1, lat_dim)
                                feat = cond["state"].view(1, -1)  # (1, feat_dim)
                                decoded = self.action_decoder.decode_action(lat_act, feat, None)  # (1, chunk*7)
                                action_venv = decoded.cpu().numpy().reshape(1, -1, 7)  # (1, chunk, 7)
                            else:
                                action_venv = samples.trajectories.cpu().numpy()[:, : self.act_steps]
                                action_venv = action_venv.reshape(1, -1, 7)
                            # Apply multi-step action
                            (
                                obs_venv,
                                reward_venv,
                                terminated_venv,
                                truncated_venv,
                                info_venv,
                            ) = self.env.step(np.squeeze(action_venv))
                            done_venv = terminated_venv | truncated_venv

                            if reward_venv > 0:
                                episode_rewards.append(reward_venv)
                            if done_venv:
                                episode_lengths.append(step)
                                break

                            # WM Encoder
                            if self.wme is not None:
                                for key in obs_venv:
                                    obs_venv[key] = np.expand_dims(obs_venv[key], 0)
                                wm_features, out_state = self.wme.get_hist_wm_features(
                                    obs_venv,
                                    action_venv,
                                    prev_done_venv,
                                    in_state,
                                    self.device,
                                )
                                prev_done_venv = np.array(done_venv).astype(bool)
                                obs_venv = wm_features.cpu().numpy()
                                in_state = out_state

                            prev_obs_venv = {"state": obs_venv}

                    avg_episode_reward = np.sum(episode_rewards) / total_episodes_eval
                    avg_episode_length = np.sum(episode_lengths) / total_episodes_eval
                    success_rate = avg_episode_reward
                    num_episode_finished = len(episode_rewards)
                    avg_best_reward = avg_episode_reward
                else:
                    raise ValueError(f"Unknown environment type: {self.env_type}")

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    obs_trajs["state"] = torch.from_numpy(obs_trajs["state"]).float().to(self.device)

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(self.n_envs * self.n_steps / self.logprob_batch_size)
                    obs_ts = [{} for _ in range(num_split)]
                    obs_k = einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                    obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                    for i, obs_t in enumerate(obs_ts_k):
                        obs_ts[i]["state"] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = self.model.critic(obs).cpu().numpy().flatten()
                        values_trajs = np.vstack((values_trajs, values.reshape(-1, self.n_envs)))
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {"state": torch.from_numpy(obs_venv["state"]).float().to(self.device)}
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = self.model.critic(obs_venv_ts).reshape(1, -1).cpu().numpy()
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = {
                    "state": einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs).float().to(self.device),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = torch.tensor(returns_trajs).float().to(self.device).reshape(-1)
                values_k = torch.tensor(values_trajs).float().to(self.device).reshape(-1)
                advantages_k = torch.tensor(advantages_trajs).float().to(self.device).reshape(-1)
                logprobs_k = torch.tensor(logprobs_trajs).float().to(self.device)

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs
                inds_k = np.arange(total_steps)
                clipfracs = []
                for update_epoch in range(self.update_epochs):
                    # for each epoch, go through all data in batches
                    flag_break = False
                    np.random.shuffle(inds_k)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        obs_b = {"state": obs_k["state"][inds_b]}
                        chains_b = chains_k[inds_b]
                        returns_b = returns_k[inds_b]
                        values_b = values_k[inds_b]
                        advantages_b = advantages_k[inds_b]
                        logprobs_b = logprobs_k[inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(),
                                    self.max_grad_norm,
                                )
                            self.actor_optimizer.step()
                            if self.learn_eta and batch % self.eta_update_interval == 0:
                                self.eta_optimizer.step()
                        self.critic_optimizer.step()
                        log.info(f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}")

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Plot state trajectories (only in D3IL)
            if self.itr % self.render_freq == 0 and self.n_render > 0 and self.traj_plotter is not None:
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_wm_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    if self.env_type == "real":
                        log.info(f"eval: WM-sim rollout at itr {self.itr}")
                        if self.use_wandb:
                            wandb.log(
                                {
                                    "total env steps - eval": cnt_wm_train_step,
                                    "itr - eval": self.itr,
                                },
                                step=self.itr,
                                commit=True,
                            )
                    else:
                        log.info(
                            f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                        )
                        if self.use_wandb:
                            wandb.log(
                                {
                                    "success rate - eval": success_rate,
                                    "avg episode reward - eval": avg_episode_reward,
                                    "avg best reward - eval": avg_best_reward,
                                    "num episode - eval": num_episode_finished,
                                    "avg episode length - eval": avg_episode_length,
                                    "total env steps - eval": cnt_wm_train_step,
                                    "itr - eval": self.itr,
                                },
                                step=self.itr,
                                commit=False,
                            )
                            if options_venv is not None and self.n_render > 0:
                                if self.n_render == 1:
                                    wandb.log(
                                        {"video - eval": wandb.Video(options_venv[rand_ind]["video_path"], format="mp4")},
                                        step=self.itr,
                                        commit=False,
                                    )
                                else:
                                    for env_ind in range(self.n_render):
                                        wandb.log(
                                            {
                                                f"video - {env_ind}": wandb.Video(
                                                    options_venv[env_ind]["video_path"], format="mp4"
                                                )
                                            },
                                            step=self.itr,
                                            commit=False,
                                        )
                            elif options_venv is not None and self.n_render == -1:
                                for env_ind in range(len(options_venv)):
                                    wandb.log(
                                        {
                                            f"video - {env_ind}": wandb.Video(
                                                options_venv[env_ind]["video_path"], format="mp4"
                                            )
                                        },
                                        step=self.itr,
                                        commit=False,
                                    )
                        run_results[-1]["eval_success_rate"] = success_rate
                        run_results[-1]["eval_episode_reward"] = avg_episode_reward
                        run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_wm_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | episode length {avg_episode_length:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        if self.itr % self.render_freq == 0:
                            if os.path.exists(train_rgb_static_path):
                                wandb.log(
                                    {"video - train rgb static": wandb.Video(train_rgb_static_path, format="mp4")},
                                    step=self.itr,
                                    commit=False,
                                )
                            if os.path.exists(train_rgb_gripper_path):
                                wandb.log(
                                    {"video - train rgb gripper": wandb.Video(train_rgb_gripper_path, format="mp4")},
                                    step=self.itr,
                                    commit=False,
                                )
                        wandb.log(
                            {
                                "total env step": cnt_wm_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "avg episode length - train": avg_episode_length,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                                "itr": self.itr,
                                # # Action-smoothness observation (continuous dims; gripper excluded).
                                "action vel mean - train": avg_action_vel,
                                "action vel max - train": _act_vel_max,
                                "action accel mean - train": avg_action_acc,
                                "action accel max - train": _act_acc_max,
                                "action jerk-penalty (sum sq accel) - train": avg_action_jerk_pen,
                                "action boundary accel mean - train": avg_action_bacc,
                                "action boundary accel max - train": _act_bacc_max,
                                "action boundary jerk-penalty (sum sq) - train": avg_action_bjerk_pen,
                                "action terminal jerk-penalty (applied) - train": avg_term_jerk_pen,
                                # Post-coefficient penalty values (= coef * raw jerk) -- the
                                # actual reward-scale magnitudes. within + boundary ~= the
                                # per-episode applied terminal penalty above.
                                "action jerk-penalty within (xcoef) - train": _act_jerk_coef * avg_action_jerk_pen,
                                "action jerk-penalty boundary (xcoef) - train": _act_jerk_coef * avg_action_bjerk_pen,
                                "action jerk coef - train": _act_jerk_coef,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
