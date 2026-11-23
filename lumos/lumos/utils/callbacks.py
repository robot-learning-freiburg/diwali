import os

import torch

from lightning.pytorch.utilities.rank_zero import rank_zero_only
import pytorch_lightning as pl

from lumos.utils.samplers import EquiDistributedSampler


class WMCustomModelCheckpoint(pl.Callback):
    def __init__(self, savepoints, dirpath, filename="{step}.ckpt"):
        self.savepoints = set(savepoints)
        self.dirpath = dirpath
        self.filename = filename
        self.lowest_val_loss = float("inf")
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in self.savepoints and trainer.global_rank == 0:
            filepath = os.path.join(self.dirpath, self.filename.format(step=step))
            trainer.save_checkpoint(filepath)
            print(f"Checkpoint saved at step {step}: {filepath}")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        filepath = os.path.join(self.dirpath, "last.ckpt")
        trainer.save_checkpoint(filepath)

        if trainer.sanity_checking:
            return
        current_val_loss = pl_module.running_metrics["loss_total"]

        if current_val_loss is not None and current_val_loss < self.lowest_val_loss:
            self.lowest_val_loss = current_val_loss.clone().detach()
            filepath = os.path.join(self.dirpath, "lowest.ckpt")
            trainer.save_checkpoint(filepath)
            print(f"New minimum val loss detected: {current_val_loss}. Checkpoint saved as: {filepath}")


class ActionDecoderCheckpoint(WMCustomModelCheckpoint):
    def __init__(self, savepoints, dirpath, filename="{step}_action_decoder.pt"):
        super().__init__(savepoints=savepoints, dirpath=dirpath, filename=filename)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in self.savepoints and trainer.global_rank == 0:
            filepath = os.path.join(self.dirpath, self.filename.format(step=step))
            self._save_action_decoder(pl_module, filepath)
            print(f"Action decoder checkpoint saved at step {step}: {filepath}")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        last_path = os.path.join(self.dirpath, "last.ckpt")
        trainer.save_checkpoint(last_path)

        if trainer.sanity_checking:
            return
        current_val_loss = pl_module.running_metrics["loss_total"]

        if current_val_loss is not None and current_val_loss < self.lowest_val_loss:
            self.lowest_val_loss = current_val_loss.clone().detach()
            filepath = os.path.join(self.dirpath, "lowest_action_decoder.pt")
            self._save_action_decoder(pl_module, filepath)
            print(
                f"New minimum val loss detected: {current_val_loss}. "
                f"Action decoder saved as: {filepath}"
            )

    def _save_action_decoder(self, pl_module, filepath: str) -> None:
        if not hasattr(pl_module, "dcd_action"):
            raise AttributeError("Expected pl_module to have dcd_action for action decoder saves.")
        torch.save(pl_module.dcd_action.state_dict(), filepath)


class SetSamplerEpochCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dataloaders = trainer.train_dataloader

        if isinstance(dataloaders, dict):
            loaders = dataloaders.values()
        else:
            loaders = [dataloaders]

        for loader in loaders:
            batch_sampler = getattr(loader, "batch_sampler", None)
            if isinstance(batch_sampler, EquiDistributedSampler):
                batch_sampler.set_epoch(trainer.current_epoch)

    def on_validation_epoch_start(self, trainer, pl_module):
        dataloaders = trainer.val_dataloaders

        if isinstance(dataloaders, dict):
            loaders = dataloaders.values()
        else:
            loaders = [dataloaders]

        for loader in loaders:
            batch_sampler = getattr(loader, "batch_sampler", None)
            if isinstance(batch_sampler, EquiDistributedSampler):
                batch_sampler.set_epoch(trainer.current_epoch)
