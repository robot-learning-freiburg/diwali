
from copy import deepcopy
import logging
from typing import Any, Dict, Optional, Tuple, Union

import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import Tensor
import torch.distributions as D

from lumos.utils.nn_utils import init_weights
from lumos.world_models.dreamer_v2_base import DreamerV2Base

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class DreamerV2(DreamerV2Base):
    """
    The lightning module used for training DreamerV2.
    Args:
    """

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        rssm: DictConfig,
        amp: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        with_proprio: bool,
        use_gripper_camera: bool,
        robot_dim: int,
        name: str,
        rebalance_loss: bool=False,
    ):
        super().__init__(amp=amp, optimizer=optimizer, train_batch_size=train_batch_size, val_batch_size=val_batch_size, with_proprio=with_proprio, use_gripper_camera=use_gripper_camera, name=name, rebalance_loss=rebalance_loss)

        self.encoder = hydra.utils.instantiate(encoder)
        decoder.in_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank
        decoder.use_gripper_camera = self.use_gripper_camera
        self.decoder = hydra.utils.instantiate(decoder)
        rssm.cell.embed_dim = encoder.cnn_depth * (2 ** (len(encoder.kernels) + 1))
        if self.with_proprio:
            rssm.cell.embed_dim += robot_dim
        self.rssm_core = hydra.utils.instantiate(rssm)

        self.kl_balance = loss.kl_balance
        self.kl_weight = loss.kl_weight
        self.image_weight = loss.image_weight
        self.act_weight = loss.act_weight
        self.grad_clip = loss.grad_clip

        self.base_metrics = [
            "loss_total",
            "loss_img",
            "loss_kl",
            "loss_kl-post",
            "loss_kl-prior",
            "entropy_prior",
            "entropy_post",
        ]
        self.batch_metrics = deepcopy(self.base_metrics)

        for m in self.modules():
            init_weights(m)
        self.save_hyperparameters()

    def forward(self, rgb_s: Tensor, act: Tensor, reset: Tensor, in_state: Tensor, rgb_g: Tensor = None, proprio: Tensor = None) -> Dict[str, Tensor]:
        embed = self.encoder(rgb_s, rgb_g)
        if self.with_proprio:
            embed = torch.cat((embed, proprio), -1)

        prior, post, features, out_states = self.rssm_core.forward(embed[1:], act, reset, in_state)

        dcd_img_s, dcd_img_g = self.decoder(features)

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_img_s": dcd_img_s,
            "dcd_img_g": dcd_img_g,
            "out_states": out_states,
        }
        return outputs

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # KL divergence loss
        dpost = self.rssm_core.zdistr(outs["post"])
        dprior = self.rssm_core.zdistr(outs["prior"])
        loss_kl_post = D.kl.kl_divergence(dpost, self.rssm_core.zdistr(outs["prior"].detach()))
        loss_kl_prior = D.kl.kl_divergence(self.rssm_core.zdistr(outs["post"].detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior

        # Image reconstruction loss
        obs = batch["rgb_obs"]["rgb_static"][1:]
        dcd_img = outs["dcd_img_s"]
        if self.use_gripper_camera:
            obs = torch.cat((batch["rgb_obs"]["rgb_static"], batch["rgb_obs"]["rgb_gripper"]), dim=2)[1:]
            dcd_img = torch.cat((outs["dcd_img_s"], outs["dcd_img_g"]), dim=2)
        loss_reconstr = 0.5 * torch.square(dcd_img - obs).sum(dim=[-1, -2, -3])

        loss = self.kl_weight * loss_kl + self.image_weight * loss_reconstr
        metrics = {
            "loss_total": loss,
            "loss_img": loss_reconstr,
            "loss_kl": loss_kl,
            "loss_kl-post": loss_kl_post,
            "loss_kl-prior": loss_kl_prior,
            "entropy_prior": dprior.entropy(),
            "entropy_post": dpost.entropy(),
        }
        if not self.flag_single_robot:
            self.add_loss_with_robot_types(metrics, batch)
        metrics = {k: v.mean() if not k.startswith("sample_count") else v for k, v in metrics.items()}

        return metrics

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
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
                # proprio=batch["robot_obs"],
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
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
                # proprio=batch["robot_obs"],
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        for key in self.running_metrics.keys():
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