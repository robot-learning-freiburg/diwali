
import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lumos.datasets.feats_dataset import FeatsDataset
from lumos.utils.nn_utils import transpose_collate_feats

logger = logging.getLogger(__name__)


class FeatsDataModule(pl.LightningDataModule):
    """LightningDataModule that loads precomputed features from features.pkl files.

    Args:
        train_feats_path: Path to training features.pkl.
        val_feats_path: Path to validation features.pkl.
        seq_len: Sequence length per sample.
        train_batch_size: Batch size for training.
        val_batch_size: Batch size for validation.
        num_workers: DataLoader workers.
        with_features: Whether to include WM features (set False for lat-only models).
        action_space: Declared for config resolution (${datamodule.action_space}).
    """

    def __init__(
        self,
        train_feats_path: str,
        val_feats_path: str,
        train_index_file: str,
        val_index_file: str,
        seq_len: int,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int = 8,
        with_features: bool = True,
        action_space: int = 56,
    ):
        super().__init__()
        self.train_feats_path = train_feats_path
        self.val_feats_path = val_feats_path
        self.train_index_file = train_index_file
        self.val_index_file = val_index_file
        self.seq_len = seq_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.with_features = with_features
        self.action_space = action_space

    def setup(self, stage=None):
        self.train_dataset = FeatsDataset(self.train_feats_path, self.train_index_file, self.seq_len, self.with_features)
        self.val_dataset = FeatsDataset(self.val_feats_path, self.val_index_file, self.seq_len, self.with_features)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transpose_collate_feats,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=transpose_collate_feats,
            persistent_workers=self.num_workers > 0,
        )
