"""Generate binary success labels for real-world expert demonstrations.

Unlike CALVIN, real-world skills have no privileged scene state to roll out
in simulation. This script auto-labels the last `n_success_frames` of each
episode as success (1) and all earlier frames as unsuccessful (0).

Output format matches `generate_rewards_calvin.py`:
    {output_dir}/{skill}/{split}_rewards.npz  with key `rewards` (int64, 0/1)
"""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


def last_n_frame_rewards(traj_lengths, n_success_frames):
    """Return a stitched 0/1 reward array aligned with concatenated episodes."""
    traj_lengths = np.asarray(traj_lengths, dtype=np.int64)
    if traj_lengths.ndim != 1:
        raise ValueError(f"traj_lengths must be 1D, got shape {traj_lengths.shape}")
    if np.any(traj_lengths <= 0):
        raise ValueError("traj_lengths must be positive")

    rewards = np.zeros(int(traj_lengths.sum()), dtype=np.int64)
    ends = np.cumsum(traj_lengths)
    starts = np.concatenate(([0], ends[:-1]))
    n_success_frames = int(n_success_frames)
    for start, end in zip(starts, ends):
        n_pos = min(n_success_frames, int(end - start))
        if n_pos > 0:
            rewards[end - n_pos : end] = 1
    return rewards


@hydra.main(version_base="1.3", config_path="../../config/rewcls", config_name="generate_rewards_real")
def main(cfg: DictConfig):
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    n_success_frames = int(cfg.n_success_frames)
    if n_success_frames <= 0:
        raise ValueError(f"n_success_frames must be positive, got {n_success_frames}")

    for split in tqdm(list(cfg.splits), desc="Split"):
        print(f"Processing split: {split}")
        for skill_name in tqdm(list(cfg.skills_list), desc="Skill"):
            data_path = input_dir / skill_name / f"{split}.npz"
            if not data_path.exists():
                print(f"Skipping {skill_name}/{split}: {data_path} does not exist.")
                continue

            data = np.load(data_path, allow_pickle=True)
            if "traj_lengths" not in data.files:
                raise KeyError(f"{data_path} has no 'traj_lengths' key. Found: {data.files}")

            traj_lengths = data["traj_lengths"]
            rewards = last_n_frame_rewards(traj_lengths, n_success_frames)

            n_frames = int(traj_lengths.sum())
            n_success = int(rewards.sum())
            n_short = int(np.sum(traj_lengths < n_success_frames))
            print(
                f"{skill_name}/{split}: {len(traj_lengths)} episodes, "
                f"{n_frames} frames, {n_success} success "
                f"({n_success / max(n_frames, 1):.1%})"
                + (f", {n_short} episodes shorter than {n_success_frames}" if n_short else "")
            )

            skill_out_dir = output_dir / skill_name
            skill_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = skill_out_dir / f"{split}_rewards.npz"
            np.savez(out_path, rewards=rewards)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
