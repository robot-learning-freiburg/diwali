import hydra
from omegaconf import DictConfig
from typing import Tuple, Optional, Dict

import torch
from torch import Tensor
from torch import amp

from diwali.wm.models.world_model import WorldModel


class DreamerV2Base(WorldModel):
    def __init__(
        self,
        amp: DictConfig,
        optimizer: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        with_proprio: bool,
        use_gripper_camera: bool,
        name: str,
        rebalance_loss: bool=False,
        scheduler: DictConfig=None,
    ):
        super().__init__(name=name)
        self.autocast = hydra.utils.instantiate(amp.autocast)
        self.scaler = hydra.utils.instantiate(amp.scaler)
        self.rebalance_loss = rebalance_loss
        
        self.with_proprio = with_proprio
        self.use_gripper_camera = use_gripper_camera

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.automatic_optimization = False
        self.num_val_batches = 0
        self.batch_metrics = []

    @torch.inference_mode()
    def infer_features(self, rgb_s: Tensor, actions: Tensor, reset: Tensor, in_state: Tensor, rgb_g: Tensor = None, proprio: Tensor = None) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            rgb_s_dev = rgb_s.to(self.device)
            rgb_g_dev = rgb_g.to(self.device) if rgb_g is not None else None
            proprio_dev = proprio.to(self.device) if self.with_proprio else None
            actions_dev = actions.to(self.device)
            reset_dev = reset.to(self.device)

            outs = self(rgb_s_dev, actions_dev, reset_dev, in_state, rgb_g=rgb_g_dev, proprio=proprio_dev)
        return outs["features"], outs["out_states"]

    def dream(self, act: Tensor, in_state: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
        with self.autocast:
            pp, (h, z) = self.rssm_core.cell.forward(act, in_state, temperature=temperature)
        return pp, (h, z)

    def pred_img(self, prior: Tensor, features: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        with torch.no_grad():
            prior_samples = self.rssm_core.zdistr(prior).sample()
            prior_samples = prior_samples.reshape(prior_samples.shape[0], prior_samples.shape[1], -1)
            features_prior = self.rssm_core.feature_replace_z(features, prior_samples)
            dcd_img_s, dcd_img_g = self.decoder(features_prior)
            return dcd_img_s, dcd_img_g

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        self.flag_single_robot = self.trainer.datamodule.single_robot
        if self.flag_single_robot:
            return
        for metric in self.base_metrics:
            self.batch_metrics += [metric + "-" + robot_name for robot_name in self.trainer.datamodule.robot_name_ids.values()]

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.in_state = self.rssm_core.init_state(self.train_batch_size)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.in_state = self.rssm_core.init_state(self.val_batch_size)
        self.running_metrics = {metric_name: 0 for metric_name in self.batch_metrics}
        if not self.flag_single_robot:
            self.running_metrics.update({f"sample_count-{robot_name}": 0 for robot_name in self.trainer.datamodule.robot_name_ids.values()})
        self.num_val_batches = 0

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        optim = {"optimizer": optimizer}
        if hasattr(self, "scheduler") and self.scheduler is not None:
            scheduler = hydra.utils.instantiate(self.scheduler, optimizer=optimizer)
            optim["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}
        return optim

    def add_loss_with_robot_types(self, metrics: Dict[str, Tensor], batch: Dict[str, Tensor]) -> None:
        """This function has 2 purposes:
            1) add robot type specific loss to metrics for better monitoring
            2) rebalance the loss to counter the sample imbalance"""
        sub_loss = list() # for rebalance with data ratio
        for idx in range(self.trainer.datamodule.robot_nums):
            robot_mask = (batch["robot_id"] == idx).flatten()
            sample_count = robot_mask.sum()
            metrics[f"sample_count-{self.trainer.datamodule.robot_name_ids[idx]}"] = sample_count.float()
            if sample_count == 0:
                continue
            sub_loss.append(metrics["loss_total"][:, robot_mask].mean()) # loss rebalance, can also add scaling later on
            for k in self.base_metrics:
                with torch.no_grad():
                    metrics[f"{k}-{self.trainer.datamodule.robot_name_ids[idx]}"] = metrics[k][:, robot_mask].mean()
        if self.rebalance_loss:
            metrics["loss_total"] = torch.stack(sub_loss).mean()

    def loss_action_postfix(self, metrics: Dict[str, Tensor], batch: Dict[str, Tensor]) -> None:
        """This function removes the action loss that should have been masked, from the total loss.
        Note: This removes only the value not the gradient, so valid only for validation."""
        total_types = 0
        sub_action_loss = 0
        for idx in range(self.trainer.datamodule.robot_nums):
            if metrics[f"sample_count-{self.trainer.datamodule.robot_name_ids[idx]}"] > 0:
                total_types += 1
            if self.trainer.datamodule.pseudo_masked_val_act[idx]:
                if self.rebalance_loss:
                    sub_action_loss += metrics[f"loss_act-{self.trainer.datamodule.robot_name_ids[idx]}"]
                else:
                    sub_action_loss += metrics[f"loss_act-{self.trainer.datamodule.robot_name_ids[idx]}"] * metrics[f"sample_count-{self.trainer.datamodule.robot_name_ids[idx]}"]
        if self.rebalance_loss:
            metrics["loss_total"] -= self.act_weight * sub_action_loss / total_types
        else:
            metrics["loss_total"] -= self.act_weight * sub_action_loss / batch["robot_id"].shape[0]

    @torch.no_grad()
    def action_metrics(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

        gt_actions = batch["actions"]["pre_actions"]
        predicted_actions = outs["pred_act"]

        finite_mask = torch.isfinite(gt_actions).all(dim=-1)
        safe_gt_actions = torch.where(finite_mask.unsqueeze(-1), gt_actions, predicted_actions)
        action_errors = torch.abs(predicted_actions - safe_gt_actions)
        
        metrics = dict()
        for dim, name in enumerate(action_names):
            if finite_mask.sum() > 0:
                pred_valid = predicted_actions[..., dim][finite_mask]
                gt_valid = gt_actions[..., dim][finite_mask]
                # action error
                metrics[name] = action_errors[..., dim][finite_mask].mean()
                # range ratio
                metrics[name + "_range_ratio"] = (pred_valid.max() - pred_valid.min()) / (gt_valid.max() - gt_valid.min() + 1e-8)
            
            if finite_mask.sum() > 1:
                # correlation coefficient
                pred_err = pred_valid - pred_valid.mean()
                gt_err = gt_valid - gt_valid.mean()
                metrics[name + "_corr"] = (pred_err * gt_err).sum() / (torch.sqrt((pred_err**2).sum() * (gt_err**2).sum()) + 1e-8)

            if not self.flag_single_robot:
                for idx in range(self.trainer.datamodule.robot_nums):
                    robot_mask = (batch["robot_id"] == idx).flatten() & finite_mask
                    robot_name = self.trainer.datamodule.robot_name_ids[idx]
                    
                    sample_count = robot_mask.sum()
                    if sample_count > 0:
                        pred_valid_rob = predicted_actions[..., dim][robot_mask]
                        gt_valid_rob = gt_actions[..., dim][robot_mask]

                        metrics[name + "-" + robot_name] = (action_errors[..., dim] * robot_mask.float()).sum() / sample_count
                        metrics[name + "_range_ratio-" + robot_name] = (pred_valid_rob.max() - pred_valid_rob.min()) / (gt_valid_rob.max() - gt_valid_rob.min() + 1e-8)
                    if sample_count > 1:
                        pred_err_rob = pred_valid_rob - pred_valid_rob.mean()
                        gt_err_rob = gt_valid_rob - gt_valid_rob.mean()
                        metrics[name + "_corr-" + robot_name] = (pred_err_rob * gt_err_rob).sum() / (torch.sqrt((pred_err_rob**2).sum() * (gt_err_rob**2).sum()) + 1e-8)
        return metrics

    @torch.no_grad()
    def log_action_metrics(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor], mode: str):
        """
        Log detailed per-dimension action metrics for better monitoring.

        Args:
            batch: Input batch containing ground truth actions
            outs: Model outputs containing predicted actions
            mode: "train" or "val"
        """
        metrics = self.action_metrics(batch, outs)
        for key, val in metrics.items():
            self.log(f"actions/{key}-{mode}", val.detach(), sync_dist=True)