
import logging
from typing import Dict, Optional

import hydra
from omegaconf import DictConfig
import torch
from torch import Tensor

from lumos.world_models.dreamer_v2_base import DreamerV2Base

logger = logging.getLogger(__name__)

_ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


class ActionDecoderFromFeats(DreamerV2Base):
    """Train action decoder from precomputed features (lat_actions + features from features.pkl).

    Unlike ActionDecoderOnly, this class requires no world model at training time.
    lat_actions and features are read directly from a precomputed features.pkl cache.
    """

    def __init__(
        self,
        dcd_action: DictConfig,
        amp: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        latent_action_dim: int,
        feature_dim: int,
        name: str,
        robot_obs_dim: int = 0,
        scheduler: DictConfig = None,
        normalize_inputs: bool = True,
    ):
        super().__init__(
            amp=amp,
            optimizer=optimizer,
            scheduler=scheduler,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            with_proprio=False,
            use_gripper_camera=False,
            name=name,
        )

        self.latent_action_dim = latent_action_dim
        self.feature_dim = feature_dim
        self.robot_obs_dim = robot_obs_dim
        self.normalize_inputs = normalize_inputs
        cond_dim = latent_action_dim + feature_dim + self.robot_obs_dim
        dcd_action.in_dim = cond_dim
        if hasattr(dcd_action, "network") and hasattr(dcd_action.network, "cond_dim"):
            dcd_action.network.cond_dim = cond_dim
        self.dcd_action = hydra.utils.instantiate(dcd_action)

        self.ln_lat = torch.nn.LayerNorm(latent_action_dim) if normalize_inputs and latent_action_dim > 0 else None
        self.ln_feat = torch.nn.LayerNorm(feature_dim) if normalize_inputs and feature_dim > 0 else None
        self.ln_rob = torch.nn.LayerNorm(self.robot_obs_dim) if normalize_inputs and self.robot_obs_dim > 0 else None

        self.act_weight = loss.act_weight
        self.grad_clip = loss.grad_clip

        self.save_hyperparameters()

    def setup(self, stage=None) -> None:
        self.flag_single_robot = True  # FeatsDataModule has no multi-robot support

    def on_train_epoch_start(self) -> None:
        pass  # no rssm_core state to initialize

    def on_validation_epoch_start(self) -> None:
        self.running_metrics = dict()
        self.action_metrics_sum = dict()
        self.num_val_batches = 0

    def _maybe_norm(self, norm, x: Tensor) -> Tensor:
        return norm(x) if norm is not None else x

    def _build_cond(self, batch: Dict[str, Tensor]) -> Tensor:
        parts = []
        if self.latent_action_dim > 0:
            parts.append(self._maybe_norm(self.ln_lat, batch["lat_actions"]))
        if self.feature_dim > 0:
            parts.append(self._maybe_norm(self.ln_feat, batch["features"]))
        if self.robot_obs_dim > 0 and "robot_obs" in batch:
            parts.append(self._maybe_norm(self.ln_rob, batch["robot_obs"]))
        return torch.cat(parts, dim=-1)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pred_act, _ = self.dcd_action.get_action(self._build_cond(batch))
        return {"pred_act": pred_act}

    def _gt_actions(self, batch: Dict[str, Tensor]) -> Tensor:
        return batch["rel_actions_chunk"]

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        gt_actions = self._gt_actions(batch)
        predicted_actions = outs["pred_act"]
        valid_mask = torch.isfinite(gt_actions).all(dim=-1, keepdim=True)
        safe_gt = torch.where(valid_mask, gt_actions, predicted_actions)
        loss_act = 0.5 * torch.square(predicted_actions - safe_gt).sum(dim=-1)
        return {"loss_total": (self.act_weight * loss_act).mean()}

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        opt = self.optimizers()
        opt.zero_grad()

        with self.autocast:
            outs = self(batch)
            losses = self.loss(batch, outs)

        self.log("loss/train", losses["loss_total"], sync_dist=True)
        self.log_action_metrics(batch, outs, mode="train")
        self.scaler.scale(losses["loss_total"]).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.scaler.step(opt)
        self.scaler.update()
        return losses["loss_total"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        with self.autocast:
            outs = self(batch)
            losses = self.loss(batch, outs)

        for key, val in losses.items():
            if key not in self.running_metrics:
                self.running_metrics[key] = val.detach()
            else:
                self.running_metrics[key] += val.detach()

        for key, val in self.action_metrics(batch, outs).items():
            if key not in self.action_metrics_sum:
                self.action_metrics_sum[key] = val.detach()
            else:
                self.action_metrics_sum[key] += val.detach()

        self.num_val_batches += 1
        return losses["loss_total"]

    def on_validation_epoch_end(self) -> None:
        for val in self.running_metrics.values():
            self.log("loss/val", val / self.num_val_batches, sync_dist=True)
        for key, val in self.action_metrics_sum.items():
            self.log(f"actions/{key}-val", val / self.num_val_batches, sync_dist=True)

    @torch.no_grad()
    def action_metrics(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        gt_actions = self._gt_actions(batch)
        predicted_actions = outs["pred_act"]

        finite_mask = torch.isfinite(gt_actions).all(dim=-1)
        safe_gt = torch.where(finite_mask.unsqueeze(-1), gt_actions, predicted_actions)
        action_errors = torch.abs(predicted_actions - safe_gt)

        metrics = {}
        for dim, name in enumerate(_ACTION_NAMES):
            if finite_mask.sum() > 0:
                pred_valid = predicted_actions[..., dim][finite_mask]
                gt_valid = gt_actions[..., dim][finite_mask]
                metrics[name] = action_errors[..., dim][finite_mask].mean()
                metrics[name + "_range_ratio"] = (pred_valid.max() - pred_valid.min()) / (
                    gt_valid.max() - gt_valid.min() + 1e-8
                )
            if finite_mask.sum() > 1:
                pred_err = pred_valid - pred_valid.mean()
                gt_err = gt_valid - gt_valid.mean()
                metrics[name + "_corr"] = (pred_err * gt_err).sum() / (
                    torch.sqrt((pred_err**2).sum() * (gt_err**2).sum()) + 1e-8
                )
        return metrics

    @torch.inference_mode()
    def decode_action(self, lat_act: Tensor, features: Tensor, robot_obs: Tensor = None) -> Tensor:
        batch = {"lat_actions": lat_act, "features": features}
        if robot_obs is not None:
            batch["robot_obs"] = robot_obs
        pred_act, _ = self.dcd_action.get_action(self._build_cond(batch))
        return pred_act


class ActionDecoderFromFeatsLatOnly(ActionDecoderFromFeats):
    """Train action decoder from lat_actions only (no WM features)."""

    def __init__(
        self,
        dcd_action: DictConfig,
        amp: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        latent_action_dim: int,
        name: str,
        scheduler: DictConfig = None,
    ):
        super().__init__(
            dcd_action=dcd_action,
            amp=amp,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            latent_action_dim=latent_action_dim,
            feature_dim=0,
            name=name,
        )

    def _build_cond(self, batch: Dict[str, Tensor]) -> Tensor:
        if self.ln_lat is not None:
            return self.ln_lat(batch["lat_actions"])
        return batch["lat_actions"]

    @torch.inference_mode()
    def decode_action(self, lat_act: Tensor) -> Tensor:
        pred_act, _ = self.dcd_action.get_action(self._build_cond({"lat_actions": lat_act}))
        return pred_act
