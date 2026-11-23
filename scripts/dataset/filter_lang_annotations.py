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


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="filter_lang_annotations")
def filter_lang_annotations(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    ep_start_end_ids = np.load(data_dir / cfg.index_file)

    ann_path = data_dir / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).reshape(-1)[0]
    indx = ann["info"]["indx"]
    tasks = ann["language"]["task"]

    keep = [
        i
        for i, (s, e) in enumerate(indx)
        if np.any((ep_start_end_ids[:, 0] <= s) & (e <= ep_start_end_ids[:, 1]))
    ]
    logger.info(f"Kept {len(keep)}/{len(indx)} annotated episodes whose frames fall within {cfg.index_file}.")

    before_counts = Counter(tasks)
    after_counts = Counter(tasks[i] for i in keep)
    logger.info("Per-skill counts for skills used in pretrain configs (before -> after, % kept):")
    for skill in cfg.skills_list:
        before = before_counts.get(skill, 0)
        after = after_counts.get(skill, 0)
        pct = 100 * after / before if before else 0.0
        logger.info(f"  {skill}: {before} -> {after} ({pct:.1f}%)")

    ann_filtered = {
        "language": {
            "task": [tasks[i] for i in keep],
            "ann": [ann["language"]["ann"][i] for i in keep],
            "emb": ann["language"]["emb"][keep],
        },
        "info": {
            "episodes": ann["info"]["episodes"],
            "indx": [indx[i] for i in keep],
        },
    }

    output_path = data_dir / "lang_annotations" / cfg.output_file
    np.save(output_path, ann_filtered)
    logger.info(f"Saved filtered annotations to {output_path}")


if __name__ == "__main__":
    filter_lang_annotations()