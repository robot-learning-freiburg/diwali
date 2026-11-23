"""Extract RGB validation start frames for world-model eval.

Unlike extract_validation_starts.py (robot_obs only, used to reset the real
robot), this writes rgb_static, rgb_gripper, and robot_obs so WM-sim eval can
initialize imagination rollouts.

Only the first save_start_length frames of each validation episode are loaded.

    python scripts/dataset/extract_validation_starts_img.py
"""

if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


def images_to_chw(images: np.ndarray) -> np.ndarray:
    """(T, H, W, C) or (T, C, H, W) -> uint8 (T, C, H, W)."""
    arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D image batch, got shape {arr.shape}")
    if arr.shape[-1] in (1, 3) and arr.shape[1] not in (1, 3):
        arr = np.transpose(arr, (0, 3, 1, 2))
    return arr.astype(np.uint8)


class CALVINStartFrameExtractor:
    def __init__(
        self,
        data_dir: str,
        skill_name: str,
        data_to_extract: list,
        save_start_length: int,
        step_len: int,
        lang_ann_file: str = "auto_lang_ann.npy",
    ):
        self.data_dir = Path(data_dir)
        self.skill_name = skill_name
        self.data_to_extract = data_to_extract
        self.save_start_length = save_start_length
        self.step_len = step_len
        self.lang_ann_file = lang_ann_file
        self.episode_lookup = self.load_file_indices(self.data_dir, self.skill_name)
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        return self.get_start_frames(idx)

    def lookup_naming_pattern(self):
        it = os.scandir(self.data_dir)
        while True:
            filename = Path(next(it))
            if "npz" in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def get_episode_name(self, idx: int) -> Path:
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def get_start_frames(self, idx: int) -> Dict:
        start_file_indx, end_file_indx = self.episode_lookup[idx]
        last_idx = min(end_file_indx, start_file_indx + (self.save_start_length - 1) * self.step_len)
        frames = [
            load_npz(self.get_episode_name(file_idx))
            for file_idx in range(start_file_indx, last_idx + 1, self.step_len)
        ]
        episode = {key: np.stack([ep[key] for ep in frames]) for key in frames[0].keys()}

        batch = {}
        if "robot_obs" in self.data_to_extract:
            batch["robot_obs"] = episode["robot_obs"]
        if "rgb_static" in self.data_to_extract:
            batch["rgb_static"] = images_to_chw(episode["rgb_static"])
        if "rgb_gripper" in self.data_to_extract:
            batch["rgb_gripper"] = images_to_chw(episode["rgb_gripper"])
        return batch

    def load_file_indices(self, data_dir: Path, skill: str) -> List[Tuple[int, int]]:
        assert data_dir.is_dir()
        file_name = data_dir / "lang_annotations" / self.lang_ann_file
        data = np.load(file_name, allow_pickle=True).reshape(-1)[0]
        all_eps_idx_part_task = [i for (i, v) in enumerate(data["language"]["task"]) if v == skill]
        episode_lookup = [data["info"]["indx"][i] for i in all_eps_idx_part_task]
        logger.info(f"Found {len(episode_lookup)} demonstrations of skill {skill}.")
        return episode_lookup


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="extract_validation_starts_img")
def make_dataset(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    data_to_extract = list(cfg.fields_to_extract)
    assert "rgb_static" in data_to_extract, "rgb_static must be in fields_to_extract"
    assert "rgb_gripper" in data_to_extract, "rgb_gripper must be in fields_to_extract"
    np.random.seed(cfg.seed)

    split = "training"
    split_dir = os.path.join(cfg.input_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Input directory {split_dir} does not exist.")

    lang_ann_file = cfg.get("lang_ann_file", "auto_lang_ann.npy")
    if not isinstance(lang_ann_file, str):
        lang_ann_file = lang_ann_file.get(split, "auto_lang_ann.npy")

    for skill in tqdm(cfg.skills_list, desc="Skills"):
        logger.info(f"Skill: {skill}")
        output_dir = os.path.join(cfg.output_dir, skill)
        os.makedirs(output_dir, exist_ok=True)

        extractor = CALVINStartFrameExtractor(
            data_dir=split_dir,
            skill_name=skill,
            data_to_extract=data_to_extract,
            save_start_length=cfg.save_start_length,
            step_len=cfg.step_len,
            lang_ann_file=lang_ann_file,
        )
        if len(extractor) == 0:
            logger.warning(f"No episodes found for skill {skill} in split {split}. Skipping...")
            continue

        if cfg.n_episodes > 0 and cfg.n_episodes < len(extractor):
            ep_indices = np.random.choice(len(extractor), cfg.n_episodes, replace=False)
            logger.info(f"Extracting {cfg.n_episodes} of {len(extractor)} episodes for skill {skill}.")
        else:
            ep_indices = np.arange(len(extractor))
            logger.info(f"Extracting all {len(extractor)} episodes for skill {skill}.")

        robot_obs_starts = []
        rgb_static_starts = []
        rgb_gripper_starts = []
        for idx in tqdm(sorted(ep_indices), desc=f"Processing skill: {skill}", total=len(ep_indices)):
            episode = extractor[int(idx)]
            if "robot_obs" in data_to_extract:
                robot_obs_starts.append(episode["robot_obs"][: cfg.save_start_length])
            rgb_static_starts.append(episode["rgb_static"][: cfg.save_start_length])
            rgb_gripper_starts.append(episode["rgb_gripper"][: cfg.save_start_length])

        save_kwargs = {
            "rgb_static": np.concatenate(rgb_static_starts, axis=0),
            "rgb_gripper": np.concatenate(rgb_gripper_starts, axis=0),
        }
        if robot_obs_starts:
            save_kwargs["robot_obs"] = np.concatenate(robot_obs_starts, axis=0)

        out_path = os.path.join(output_dir, f"val_starts_{cfg.save_start_length}.npz")
        np.savez(out_path, **save_kwargs)
        logger.info(
            "Saved %s (rgb_static %s, rgb_gripper %s, robot_obs %s)",
            out_path,
            save_kwargs["rgb_static"].shape,
            save_kwargs["rgb_gripper"].shape,
            save_kwargs["robot_obs"].shape if "robot_obs" in save_kwargs else None,
        )

    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))


if __name__ == "__main__":
    make_dataset()
