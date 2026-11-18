"""
Filters `lang_annotations/auto_lang_ann.npy` down to the annotated sub-episodes
that fall within a restricted set of raw episodes (e.g. `ep_start_end_ids_franka.npy`).
"""
import logging
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="expert_to_lang_annotations")
def expert_to_lang_annotations(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    for split in ["training"]:
        ep_start_end_ids = np.load(data_dir / split / cfg.index_file)
        n_episodes = len(ep_start_end_ids)

        ann_path = data_dir / split / "lang_annotations" / cfg.target_index_file
        ann_path.parent.mkdir(parents=True, exist_ok=True)

        ann = {
            "language": {
            "task": [cfg.expert_task_name] * n_episodes,
            "ann": [cfg.expert_task_lang_prompt] * n_episodes,
            "emb": None,
            },
            "info": {
                "episodes": [],
                "indx": [(s, e) for s, e in ep_start_end_ids],
            },
        }
        np.save(ann_path, ann)
        logger.info(f"Saved expert annotations to {ann_path}")


if __name__ == "__main__":
    expert_to_lang_annotations()