
from copy import deepcopy
import logging
from typing import Any, Dict, Optional, Tuple, Union

import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import Tensor
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw

from diwali.wm.model_utils.nn_utils import init_weights, flatten_batch, unflatten_batch
from diwali.wm.model_utils.of_utils import create_flow_wheel, flow_to_rgb, warp2d, motion_smoothness
from diwali.wm.model_utils.pf_utils import resize_img, make_source_points_grid, draw_target_points, draw_source_points
from diwali.wm.models.dreamer_v2_base import DreamerV2Base

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class DreamerV2(DreamerV2Base):
    """
    The lightning module used for training DreamerV2 with optical and point flow.
    Args:
    """

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        fuser: DictConfig,
        inv_dyna: DictConfig,
        action_decoder: DictConfig,
        rssm: DictConfig,
        amp: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        with_proprio: bool,
        use_gripper_camera: bool,
        latent_action_dim: int,
        robot_dim: int,
        name: str,
        rebalance_loss: bool=False,
        cond_action_decoder_on_latent: bool = False,
    ):
        super().__init__(amp=amp, optimizer=optimizer, train_batch_size=train_batch_size, val_batch_size=val_batch_size, with_proprio=with_proprio, use_gripper_camera=use_gripper_camera, name=name, rebalance_loss=rebalance_loss)
        embed_dim = encoder.cnn_depth * (2 ** (len(encoder.kernels) + 1))
        wm_latent_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank

        # Separate encoders for static and gripper cameras like in Flower
        self.encoder_static = hydra.utils.instantiate(encoder)
        self.encoder_gripper = hydra.utils.instantiate(encoder)

        # Fuser to combine embeddings
        fuser.embed_dim = embed_dim
        self.fuser = hydra.utils.instantiate(fuser)

        # Original decoder for image reconstruction
        decoder.in_dim = wm_latent_dim
        decoder.use_gripper_camera = self.use_gripper_camera
        self.decoder = hydra.utils.instantiate(decoder)
        
        # Inverse dynamics for action prediction
        inv_dyna.in_dim = 2 * embed_dim
        self.inv_dyna = hydra.utils.instantiate(inv_dyna)

        # Action decoder: latent-action only by default so existing checkpoints
        # keep loading. Set cond_action_decoder_on_latent to also condition on
        # the RSSM state (concat [lat_act, features]).
        self.cond_action_decoder_on_latent = cond_action_decoder_on_latent
        action_decoder.in_dim = latent_action_dim
        if self.cond_action_decoder_on_latent:
            action_decoder.in_dim += wm_latent_dim
        self.action_decoder = hydra.utils.instantiate(action_decoder)

        # RSSM core with updated embedding dimension
        rssm.cell.embed_dim = embed_dim
        rssm.cell.action_dim = latent_action_dim
        if self.with_proprio:
            rssm.cell.embed_dim += robot_dim
        self.rssm_core = hydra.utils.instantiate(rssm)

        self.kl_balance = loss.kl_balance
        self.kl_weight = loss.kl_weight
        self.image_weight = loss.image_weight
        self.grad_clip = loss.grad_clip
        self.act_weight = loss.act_weight

        self.base_metrics = [
            "loss_total",
            "loss_img",
            "loss_kl",
            "loss_kl-post",
            "loss_kl-prior",
            "entropy_prior",
            "entropy_post",
            "loss_act",
        ]
        self.batch_metrics = deepcopy(self.base_metrics)

        for m in self.modules():
            init_weights(m)
        self.save_hyperparameters()

    def _action_decoder_args(self, lat_act: Tensor, features: Tensor) -> Tuple[Tensor, ...]:
        """Inputs for action_decoder.get_action. Order is [lat_act, features] to match ActionDecoderFromFeats."""
        if self.cond_action_decoder_on_latent:
            return lat_act, features
        return (lat_act,)

    def forward(self, rgb_s: Tensor, reset: Tensor, in_state: Tensor, rgb_g: Tensor = None, proprio: Tensor = None) -> Dict[str, Tensor]:
        # Encode using separate encoders like in Flower
        embed_s, skip_features_s = self.encoder_static(rgb_s)
        if rgb_g is not None:
            embed_g, skip_features_g = self.encoder_gripper(rgb_g)
            embed = self.fuser(embed_s, embed_g)
        else:
            embed = embed_s

        if self.with_proprio:
            embed = torch.cat((embed, proprio), -1)

        # Flow prediction using consecutive embeddings
        embed_1 = embed[:-1]
        embed_2 = embed[1:]

        # Action prediction using inverse dynamics
        lat_act, entropy = self.inv_dyna.get_action(embed_1, embed_2)

        prior, post, features, out_states = self.rssm_core.forward(embed[1:], lat_act, reset, in_state)
        dcd_act, _ = self.action_decoder.get_action(*self._action_decoder_args(lat_act, features))

        # Image reconstruction
        dcd_img_s, dcd_img_g = self.decoder(features)

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_img_s": dcd_img_s,
            "dcd_img_g": dcd_img_g,
            "lat_act": lat_act,
            "dcd_act": dcd_act,
            "entropy": entropy,
            "out_states": out_states,
        }
        return outputs

    @torch.inference_mode()
    def infer_features_noact(self, rgb_s: Tensor, reset: Tensor, in_state: Tensor, rgb_g: Tensor = None, proprio: Tensor = None) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            rgb_s_dev = rgb_s.to(self.device)
            rgb_g_dev = rgb_g.to(self.device) if rgb_g is not None else None
            proprio_dev = proprio.to(self.device) if self.with_proprio else None
            reset_dev = reset.to(self.device)

            outs = self(rgb_s_dev, reset_dev, in_state, rgb_g=rgb_g_dev, proprio=proprio_dev)
        return outs["features"], outs["out_states"]

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        metrics = dict()

        # KL divergence loss
        dpost = self.rssm_core.zdistr(outs["post"])
        dprior = self.rssm_core.zdistr(outs["prior"])
        loss_kl_post = D.kl.kl_divergence(dpost, self.rssm_core.zdistr(outs["prior"].detach()))
        loss_kl_prior = D.kl.kl_divergence(self.rssm_core.zdistr(outs["post"].detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior
        metrics["loss_kl"] = loss_kl
        metrics["loss_kl-prior"] = loss_kl_prior
        metrics["loss_kl-post"] = loss_kl_post
        metrics["entropy_prior"] = dprior.entropy()
        metrics["entropy_post"] = dpost.entropy()

        # Image reconstruction loss
        obs = batch["rgb_obs"]["rgb_static"][1:]
        dcd_img = outs["dcd_img_s"]
        if self.use_gripper_camera:
            obs = torch.cat((batch["rgb_obs"]["rgb_static"], batch["rgb_obs"]["rgb_gripper"]), dim=2)[1:]
            dcd_img = torch.cat((outs["dcd_img_s"], outs["dcd_img_g"]), dim=2)
        loss_reconstr = 0.5 * torch.square(dcd_img - obs).sum(dim=[-1, -2, -3])
        metrics["loss_img"] = loss_reconstr

        # Action prediction loss
        gt_actions = batch["actions"]["pre_actions"]
        predicted_actions = outs["dcd_act"]
        valid_action_mask = torch.isfinite(gt_actions).all(dim=-1, keepdim=True)
        safe_gt_actions = torch.where(valid_action_mask, gt_actions, predicted_actions)
        loss_act = 0.5 * torch.square(predicted_actions - safe_gt_actions).sum(dim=[-1])
        metrics["loss_act"] = loss_act

        # Combine all losses
        loss = (
            self.kl_weight * loss_kl
            + self.image_weight * loss_reconstr
            + self.act_weight * loss_act
        )
        metrics["loss_total"] = loss
        if not self.flag_single_robot:
            self.add_loss_with_robot_types(metrics, batch)

        # For metrics, compute action loss mean only over valid actions
        metrics["loss_act"] = loss_act.sum() / valid_action_mask.float().sum().clamp_min(1.0)
        metrics = {k: v.mean() if not k.startswith("sample_count") else v for k, v in metrics.items()}

        return metrics

    @torch.inference_mode()
    def infer_action(self, rgb_s: Tensor, rgb_g: Tensor = None, proprio: Tensor = None) -> Tensor:
        self.eval()
        with self.autocast:
            rgb_s_dev = rgb_s.to(self.device)
            rgb_g_dev = rgb_g.to(self.device) if rgb_g is not None else None
            proprio_dev = proprio.to(self.device) if proprio is not None else None

            # Encode using separate encoders
            embed_s, _ = self.encoder_static(rgb_s_dev)
            if rgb_g_dev is not None:
                embed_g, _ = self.encoder_gripper(rgb_g_dev)
                embed = self.fuser(embed_s, embed_g)
            else:
                embed = embed_s

            if self.with_proprio and proprio_dev is not None:
                embed = torch.cat((embed, proprio_dev), -1)

            lat_act, _ = self.inv_dyna.get_action(embed[:-1], embed[1:])

        return lat_act

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        opt = self.optimizers()
        opt.zero_grad()
        batch = batch["vis"]

        # prevent non-existent key when resume training from ckpt
        if not hasattr(self, "in_state"):
            self.in_state = self.rssm_core.init_state(self.train_batch_size)

        with self.autocast:
            rgb_g_input = batch["rgb_obs"].get("rgb_gripper")
            outs = self(
                batch["rgb_obs"]["rgb_static"],
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        self.log_metrics(losses, mode="train")

        
        if self.global_step % self.trainer.log_every_n_steps == 0:
            pred_img_s, pred_img_g = self.pred_img(*samples)

            gt_img_g_val, dcd_img_g_val, pred_img_g_val = None, None, None
            if self.use_gripper_camera:
                gt_img_g_val = batch["rgb_obs"].get("rgb_gripper")[-1, 0]
                dcd_img_g_val = outs.get("dcd_img_g")[-1, 0]
                pred_img_g_val = pred_img_g[-1, 0]

            # log gt-pred
            self.log_images(
                robot_id=batch["robot_id"][-1],
                gt_img_s=batch["rgb_obs"]["rgb_static"][-1, 0],
                dcd_img_s=outs["dcd_img_s"][-1, 0],
                pred_img_s=pred_img_s[-1, 0],
                mode="train",
                gt_img_g=gt_img_g_val,
                dcd_img_g=dcd_img_g_val,
                pred_img_g=pred_img_g_val,
            )

        self.scaler.scale(losses["loss_total"]).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.scaler.step(opt)
        self.scaler.update()

        return losses["loss_total"]

    @torch.inference_mode()
    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        with self.autocast:
            rgb_g_input = batch["rgb_obs"].get("rgb_gripper")
            outs = self(
                batch["rgb_obs"]["rgb_static"],
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
                # proprio=batch["robot_obs"],
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        for key in losses.keys():
            self.running_metrics[key] += losses[key]
        self.num_val_batches += 1

        # keep track of last batch for logging
        self.val_robot_id = batch["robot_id"][-1]
        self.val_gt_img_s = batch["rgb_obs"]["rgb_static"][-1, 0]
        self.val_dcd_img_s = outs["dcd_img_s"][-1, 0]

        self.val_gt_img_g, self.val_dcd_img_g = None, None
        if self.use_gripper_camera:
            self.val_gt_img_g = rgb_g_input[-1, 0]
            self.val_dcd_img_g = outs.get("dcd_img_g")[-1, 0]

        self.val_samples = samples
        self.last_val_batch = batch
        self.last_val_outs = outs
        return losses["loss_total"]

    @torch.inference_mode()
    def on_validation_epoch_end(self) -> None:
        for key in self.running_metrics.keys():
            if not key.startswith("sample_count"):
                self.running_metrics[key] /= self.num_val_batches
        self.log_metrics(self.running_metrics, mode="val", sync_dist=True)

        pred_img_s, pred_img_g = self.pred_img(*self.val_samples)
        pred_img_g_val = None
        if self.use_gripper_camera:
            pred_img_g_val = pred_img_g[-1, 0]

        self.log_images(
            robot_id=self.val_robot_id,
            gt_img_s=self.val_gt_img_s,
            dcd_img_s=self.val_dcd_img_s,
            pred_img_s=pred_img_s[-1, 0],
            mode="val",
            gt_img_g=self.val_gt_img_g,
            dcd_img_g=self.val_dcd_img_g,
            pred_img_g=pred_img_g_val,
        )
