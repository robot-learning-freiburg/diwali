if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import logging
import os
from pathlib import Path
import pickle
import re
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


split_to_file = {
    "training": "train",
    "validation": "val",
}


class CALVINSkillExtractor:
    """
    This class is used to extract skills from the featurised CALVIN dataset
    (but it also accesses the raw CALVIN dataset). An object of this is iterable
    and returns one episode of the chosen skill as a dictionary.
    """

    def __init__(
        self,
        data_dir: str,
        skill_name: str,
        data_to_extract: list,
        step_len: int,
        features_file_path: str,
        lang_ann_file: str = "auto_lang_ann.npy",
    ):
        self.data_dir = Path(data_dir)
        self.skill_name = skill_name
        self.data_to_extract = data_to_extract
        self.lang_ann_file = lang_ann_file
        self.episode_lookup = self.load_file_indices(self.data_dir, self.skill_name)
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()
        self.step_len = step_len
        feature_pkl = Path(features_file_path)
        assert feature_pkl.exists(), f"Feature file {feature_pkl} does not exist."
        self.features = pickle.load(open(feature_pkl, "rb"))

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        return self.get_sequences(idx)

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
        """
        Convert frame idx to file name
        """
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def zip_sequence(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame
        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        episodes = [
            load_npz(self.get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx + 1, self.step_len)
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key, _ in episodes[0].items()}
        return episode

    def zip_sequence_feat(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame
        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        episodes = [self.features[file_idx] for file_idx in range(start_idx, end_idx + 1, self.step_len)]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key, _ in episodes[0].items()}
        return episode

    def get_sequences(self, idx: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """
        info_indx = self.episode_lookup[idx]
        start_file_indx = info_indx[0]
        end_file_indx = info_indx[1]

        episode = self.zip_sequence_feat(start_file_indx, end_file_indx)
        episode_raw = self.zip_sequence(start_file_indx, end_file_indx)

        batch = {}
        if "features" in self.data_to_extract:
            batch.update({"features": episode["features"]})
        if "rel_actions" in self.data_to_extract:
            batch.update({"rel_actions": episode["rel_actions_chunk"]})
        if "lat_actions" in episode:
            batch["lat_actions"] = episode["lat_actions"]
        if "robot_obs" in episode:
            batch["robot_obs"] = episode["robot_obs"]
        # raw robot_obs and scene_obs for env.reset in sim tests
        batch["robots"] = episode_raw["robot_obs"]
        batch["scenes"] = episode_raw["scene_obs"]

        return batch

    def load_file_indices(self, data_dir: Path, skill: str) -> Tuple[List, List]:
        """
        this method builds the mapping from index to file_name used for loading the episodes
        parameters
        ----------
        data_dir:               absolute path of the directory containing the datasets
        returns
        ----------
        episode_lookup:                 list for the mapping from training example index to episode (file) index
        max_batched_length_per_demo:    list of possible starting indices per episode
        """
        assert data_dir.is_dir()
        skill_name = skill

        episode_lookup = []

        file_name = data_dir / "lang_annotations" / self.lang_ann_file
        data = np.load(file_name, allow_pickle=True).reshape(-1)[0]

        all_eps_idx_part_task = [i for (i, v) in enumerate(data["language"]["task"]) if v == skill_name]
        all_eps_start_end_part_task = [data["info"]["indx"][i] for i in all_eps_idx_part_task]

        for i in range(len(all_eps_start_end_part_task)):
            episode_lookup.append(all_eps_start_end_part_task[i])

        logger.info(f"Found {len(episode_lookup)} demonstrations of skill {skill_name}.")
        return episode_lookup


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="extract_expert_demos_feat")
def make_dataset(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    skills_list = cfg.skills_list
    data_to_extract = cfg.fields_to_extract
    assert "features" in data_to_extract, "features must be included in the fields to extract."
    assert "rel_actions" in data_to_extract, "rel_actions must be included in the fields to extract."
    
    use_decoder = getattr(cfg, "decoder_ckpt_path", None) is not None
    if use_decoder:
        from diwali.wm.models.action_decoder_from_feats import ActionDecoderFromFeats
        decoder_device = getattr(cfg, "decoder_device", "cpu")
        decoder = ActionDecoderFromFeats.load_from_checkpoint(cfg.decoder_ckpt_path, map_location="cpu", weights_only=False)
        decoder.to(decoder_device).eval()
        logger.info(f"Loaded action decoder from {cfg.decoder_ckpt_path} on {decoder_device}")

    np.random.seed(cfg.seed)
    for split in ["training"]:
        split_dir = os.path.join(cfg.input_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Input directory {split_dir} does not exist.")
        file_name = split_to_file[split]
        logger.info(f"Processing split: {split}")

        lang_ann_file = cfg.get("lang_ann_file", "auto_lang_ann.npy")
        if not isinstance(lang_ann_file, str):
            lang_ann_file = lang_ann_file.get(split, "auto_lang_ann.npy")

        for skill in tqdm(skills_list):
            logger.info(f"Skill: {skill}")

            output_dir = os.path.join(cfg.output_dir, skill)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            extractor = CALVINSkillExtractor(
                data_dir=split_dir,
                skill_name=skill,
                data_to_extract=data_to_extract,
                step_len=cfg.step_len,
                features_file_path=cfg.features_file_path,
                lang_ann_file=lang_ann_file,
            )

            if len(extractor) == 0:
                logger.warning(f"No episodes found for skill {skill} in split {split}. Skipping...")
                continue

            states = np.array([])
            actions = np.array([])
            decoded_actions_arr = np.array([])
            lat_actions_arr = np.array([])
            robot_obs_arr = np.array([])
            robots_arr = np.array([])
            scenes_arr = np.array([])
            traj_lengths = np.array([])

            if cfg.n_episodes < len(extractor) and cfg.n_episodes > 0 and "val" not in split:
                # randomly sample a subset of episodes
                ep_indices = np.random.choice(len(extractor), cfg.n_episodes, replace=False)
                logger.info(
                    f"Extracting only {cfg.n_episodes} episodes for skill {skill} from {len(extractor)} total episodes."
                )
                # save episode indices for in a text file for quick access
                with open(os.path.join(output_dir, f"{file_name}_ep_indices.txt"), "w") as f:
                    for idx in sorted(ep_indices):
                        f.write(f"{idx}\n")
            else:
                # use all episodes
                ep_indices = np.arange(len(extractor))
                logger.info(f"Extracting all {len(extractor)} episodes for skill {skill}.")

            for idx in tqdm(ep_indices, desc=f"Processing skill: {skill}"):
                episode = extractor[idx]
                rel_act = episode["rel_actions"]
                eps_len = int(rel_act.shape[0])

                features = episode["features"]
                states = np.vstack((states, features)) if states.size else features
                actions = np.vstack((actions, rel_act)) if actions.size else rel_act

                if use_decoder:
                    lat = torch.tensor(episode["lat_actions"], dtype=torch.float32).to(decoder_device)
                    feat = torch.tensor(episode["features"], dtype=torch.float32).to(decoder_device)
                    rob = torch.tensor(episode["robot_obs"], dtype=torch.float32).to(decoder_device) if "robot_obs" in episode and decoder.robot_obs_dim > 0 else None
                    with torch.no_grad():
                        dcd_acts = decoder.decode_action(lat, feat, rob).cpu().numpy()  # (T, action_space)
                    decoded_actions_arr = np.vstack((decoded_actions_arr, dcd_acts)) if decoded_actions_arr.size else dcd_acts

                if "lat_actions" in episode:
                    lat_actions_arr = np.vstack((lat_actions_arr, episode["lat_actions"])) if lat_actions_arr.size else episode["lat_actions"]

                if "robot_obs" in episode:
                    robot_obs_arr = np.vstack((robot_obs_arr, episode["robot_obs"])) if robot_obs_arr.size else episode["robot_obs"]

                robots = episode["robots"][:eps_len]
                robots_arr = np.vstack((robots_arr, robots)) if robots_arr.size else robots

                scenes = episode["scenes"][:eps_len]
                scenes_arr = np.vstack((scenes_arr, scenes)) if scenes_arr.size else scenes

                traj_lengths = np.append(traj_lengths, eps_len)

            save_dict = dict(
                states=states,
                actions=actions,
                traj_lengths=traj_lengths.astype(int),
                robots=robots_arr,
                scenes=scenes_arr,
            )
            if decoded_actions_arr.size:
                save_dict["dcd_actions"] = decoded_actions_arr
            if lat_actions_arr.size:
                save_dict["lat_actions"] = lat_actions_arr
            if robot_obs_arr.size:
                save_dict["robot_obs"] = robot_obs_arr
            # save in robomimic format
            np.savez(os.path.join(output_dir, f"{file_name}.npz"), **save_dict)
    # Save the config file for reproducibility
    OmegaConf.save(
        cfg,
        os.path.join(cfg.output_dir, "config.yaml"),
    )


if __name__ == "__main__":
    make_dataset()