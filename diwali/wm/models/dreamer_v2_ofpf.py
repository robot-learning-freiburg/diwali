
from copy import deepcopy
import logging
from typing import Any, Dict, Optional, Tuple, Union

import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import Tensor
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw

from diwali.wm.model_utils.nn_utils import init_weights, flatten_batch, unflatten_batch
from diwali.wm.model_utils.of_utils import create_flow_wheel, flow_to_rgb, warp2d, motion_smoothness
from diwali.wm.model_utils.pf_utils import resize_img, make_source_points_grid, draw_target_points, draw_source_points
from diwali.wm.models.dreamer_v2_base import DreamerV2Base

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class DreamerV2(DreamerV2Base):
    """
    The lightning module used for training DreamerV2 with optical and point flow.
    Args:
    """

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        fuser: DictConfig,
        of_decoder: DictConfig,
        pf_decoder: DictConfig,
        inv_dyna: DictConfig,
        rssm: DictConfig,
        amp: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        with_proprio: bool,
        use_gripper_camera: bool,
        latent_action_dim: int,
        robot_dim: int,
        name: str,
        rebalance_loss: bool=False,
    ):
        super().__init__(amp=amp, optimizer=optimizer, train_batch_size=train_batch_size, val_batch_size=val_batch_size, with_proprio=with_proprio, use_gripper_camera=use_gripper_camera, name=name, rebalance_loss=rebalance_loss)
        embed_dim = encoder.cnn_depth * (2 ** (len(encoder.kernels) + 1))

        # Separate encoders for static and gripper cameras like in Flower
        self.encoder_static = hydra.utils.instantiate(encoder)
        self.encoder_gripper = hydra.utils.instantiate(encoder)

        # Fuser to combine embeddings
        fuser.embed_dim = embed_dim
        self.fuser = hydra.utils.instantiate(fuser)

        # Original decoder for image reconstruction
        decoder.in_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank
        decoder.use_gripper_camera = self.use_gripper_camera
        self.decoder = hydra.utils.instantiate(decoder)
        
        # Inverse dynamics for action prediction
        inv_dyna.in_dim = 2 * embed_dim
        self.inv_dyna = hydra.utils.instantiate(inv_dyna)
        
        # Latent action decoders for optical flow and points displacement prediction
        if of_decoder is not None and len(of_decoder) > 0:
            of_decoder.in_dim = embed_dim + latent_action_dim
            self.of_static = hydra.utils.instantiate(of_decoder)
            self.of_gripper = hydra.utils.instantiate(of_decoder)
            self.use_of_loss = True
        else:
            self.use_of_loss = False

        if pf_decoder is not None and len(pf_decoder) > 0:
            pf_decoder.embed_dim = embed_dim
            pf_decoder.latent_action_dim = latent_action_dim
            self.pf_static = hydra.utils.instantiate(pf_decoder)
            self.pf_gripper = hydra.utils.instantiate(pf_decoder)
            self.use_pf_loss = True
        else:
            self.use_pf_loss = False

        # RSSM core with updated embedding dimension
        rssm.cell.embed_dim = embed_dim
        rssm.cell.action_dim = latent_action_dim
        if self.with_proprio:
            rssm.cell.embed_dim += robot_dim
        self.rssm_core = hydra.utils.instantiate(rssm)

        self.kl_balance = loss.kl_balance
        self.kl_weight = loss.kl_weight
        self.image_weight = loss.image_weight
        self.of_weight = loss.of_weight
        self.of_gripper_weight = 0
        self.of_smooth_weight = loss.of_smooth_weight
        self.of_sg_ratio = loss.of_sg_ratio
        self.pf_weight = loss.pf_weight
        self.pf_static_weight = loss.pf_static_weight
        self.pf_gripper_weight = loss.pf_gripper_weight
        self.grad_clip = loss.grad_clip

        self.base_metrics = [
            "loss_total",
            "loss_img",
            "loss_kl",
            "loss_kl-post",
            "loss_kl-prior",
            "loss_of",
            "loss_of-smooth",
            "loss_pf-s",
            "loss_pf-g",
            "loss_pf",
            "entropy_prior",
            "entropy_post",
        ]
        self.batch_metrics = deepcopy(self.base_metrics)

        for m in self.modules():
            init_weights(m)
        self.save_hyperparameters()

    def forward(self, rgb_s: Tensor, reset: Tensor, in_state: Tensor, rgb_g: Tensor = None, proprio: Tensor = None) -> Dict[str, Tensor]:
        # Encode using separate encoders like in Flower
        embed_s, skip_features_s = self.encoder_static(rgb_s)
        if rgb_g is not None:
            embed_g, skip_features_g = self.encoder_gripper(rgb_g)
            embed = self.fuser(embed_s, embed_g)
        else:
            embed = embed_s

        if self.with_proprio:
            embed = torch.cat((embed, proprio), -1)

        # Flow prediction using consecutive embeddings
        embed_1 = embed[:-1]
        embed_2 = embed[1:]

        # Action prediction using inverse dynamics
        lat_act, entropy = self.inv_dyna.get_action(embed_1, embed_2)

        if self.use_of_loss:
            of_s = self.of_static(embed_1.detach(), lat_act, skip_features_s)
            of_g = self.of_gripper(embed_1.detach(), lat_act, skip_features_g) if rgb_g is not None else None
        else:
            of_s = None
            of_g = None

        if self.use_pf_loss:
            pf_s = self.pf_static(embed_1.detach(), lat_act)
            pf_g = self.pf_gripper(embed_1.detach(), lat_act) if rgb_g is not None else None
        else:
            pf_s = None
            pf_g = None

        prior, post, features, out_states = self.rssm_core.forward(embed[1:], lat_act, reset, in_state)

        # Image reconstruction
        dcd_img_s, dcd_img_g = self.decoder(features)

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_img_s": dcd_img_s,
            "dcd_img_g": dcd_img_g,
            "of_s": of_s,
            "of_g": of_g,
            "pf_s": pf_s,
            "pf_g": pf_g,
            "lat_act": lat_act,
            "entropy": entropy,
            "out_states": out_states,
        }
        return outputs

    @torch.inference_mode()
    def infer_features_noact(self, rgb_s: Tensor, reset: Tensor, in_state: Tensor, rgb_g: Tensor = None, proprio: Tensor = None) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            rgb_s_dev = rgb_s.to(self.device)
            rgb_g_dev = rgb_g.to(self.device) if rgb_g is not None else None
            proprio_dev = proprio.to(self.device) if self.with_proprio else None
            reset_dev = reset.to(self.device)

            outs = self(rgb_s_dev, reset_dev, in_state, rgb_g=rgb_g_dev, proprio=proprio_dev)
        return outs["features"], outs["out_states"]

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        metrics = dict()

        # KL divergence loss
        dpost = self.rssm_core.zdistr(outs["post"])
        dprior = self.rssm_core.zdistr(outs["prior"])
        loss_kl_post = D.kl.kl_divergence(dpost, self.rssm_core.zdistr(outs["prior"].detach()))
        loss_kl_prior = D.kl.kl_divergence(self.rssm_core.zdistr(outs["post"].detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior
        metrics["loss_kl"] = loss_kl
        metrics["loss_kl-prior"] = loss_kl_prior
        metrics["loss_kl-post"] = loss_kl_post
        metrics["entropy_prior"] = dprior.entropy()
        metrics["entropy_post"] = dpost.entropy()

        # Image reconstruction loss
        obs = batch["rgb_obs"]["rgb_static"][1:]
        dcd_img = outs["dcd_img_s"]
        if self.use_gripper_camera:
            obs = torch.cat((batch["rgb_obs"]["rgb_static"], batch["rgb_obs"]["rgb_gripper"]), dim=2)[1:]
            dcd_img = torch.cat((outs["dcd_img_s"], outs["dcd_img_g"]), dim=2)
        loss_reconstr = 0.5 * torch.square(dcd_img - obs).sum(dim=[-1, -2, -3])
        metrics["loss_img"] = loss_reconstr

        # Optical flow loss
        if self.use_of_loss:
            loss_of, loss_of_smooth = torch.zeros_like(loss_reconstr), torch.zeros_like(loss_reconstr)
            if self.of_weight > 0.0:
                loss_of = self.of_cns_loss(batch["rgb_obs"]["rgb_static"], outs["of_s"])
                loss_of_smooth = self.of_smoothness_loss(batch["rgb_obs"]["rgb_static"], outs["of_s"])
                if self.use_gripper_camera and self.of_gripper_weight > 0.0:
                    loss_of_g = self.of_cns_loss(batch["rgb_obs"]["rgb_gripper"], outs["of_g"])
                    loss_of = loss_of * self.of_sg_ratio + loss_of_g * (1 - self.of_sg_ratio)
                    loss_of_smooth_g = self.of_smoothness_loss(batch["rgb_obs"]["rgb_gripper"], outs["of_g"])
                    loss_of_smooth = loss_of_smooth * self.of_sg_ratio + loss_of_smooth_g * (1 - self.of_sg_ratio)
                loss_of = loss_of if self.of_weight > 0.0 else torch.zeros_like(loss_reconstr)
                loss_of_smooth = loss_of_smooth if self.of_smooth_weight > 0.0 else torch.zeros_like(loss_reconstr)
        else:
            loss_of = torch.zeros_like(loss_reconstr)
            loss_of_smooth = torch.zeros_like(loss_reconstr)
        metrics["loss_of"] = loss_of
        metrics["loss_of-smooth"] = loss_of_smooth

        # Point flow loss
        if self.use_pf_loss:
            obs_size = batch["rgb_obs"]["rgb_static"][1:].shape[-1]
            loss_pf_s = self.pf_loss(batch["p_disp"]["rgb_static"], outs["pf_s"], obs_size)
            if self.use_gripper_camera and self.pf_gripper_weight > 0.0:
                loss_pf_g = self.pf_loss(batch["p_disp"]["rgb_gripper"], outs["pf_g"], obs_size)
                loss_pf = self.pf_static_weight * loss_pf_s + self.pf_gripper_weight * loss_pf_g 
            else:
                loss_pf = self.pf_static_weight * loss_pf_s
                loss_pf_g = torch.zeros_like(loss_reconstr)
        else:
            loss_pf_s = torch.zeros_like(loss_reconstr)
            loss_pf_g = torch.zeros_like(loss_reconstr)
            loss_pf = torch.zeros_like(loss_reconstr)
        metrics["loss_pf-s"] = loss_pf_s
        metrics["loss_pf-g"] = loss_pf_g
        metrics["loss_pf"] = loss_pf
        # Combine all losses
        loss = (
            self.kl_weight * loss_kl
            + self.image_weight * loss_reconstr
            + self.of_weight * loss_of
            + self.of_smooth_weight * loss_of_smooth
            + self.pf_weight * loss_pf
        )
        metrics["loss_total"] = loss
        if not self.flag_single_robot:
            self.add_loss_with_robot_types(metrics, batch)

        # For metrics, compute action loss mean only over valid actions
        metrics = {k: v.mean() if not k.startswith("sample_count") else v for k, v in metrics.items()}

        return metrics

    @torch.inference_mode()
    def infer_action(self, rgb_s: Tensor, rgb_g: Tensor = None, proprio: Tensor = None) -> Tensor:
        """Infer optical flow and actions from consecutive frames."""
        self.eval()
        with self.autocast:
            rgb_s_dev = rgb_s.to(self.device)
            rgb_g_dev = rgb_g.to(self.device) if rgb_g is not None else None
            proprio_dev = proprio.to(self.device) if proprio is not None else None

            # Encode using separate encoders
            embed_s, _ = self.encoder_static(rgb_s_dev)
            if rgb_g_dev is not None:
                embed_g, _ = self.encoder_gripper(rgb_g_dev)
                embed = self.fuser(embed_s, embed_g)
            else:
                embed = embed_s

            if self.with_proprio and proprio_dev is not None:
                embed = torch.cat((embed, proprio_dev), -1)

            lat_act, _ = self.inv_dyna.get_action(embed[:-1], embed[1:])

        return lat_act

    def on_fit_start(self):
        if self.use_of_loss:
            flow_wheel = create_flow_wheel(device=self.device, wheel_size=256, max_magnitude=10.0)
            self.log_image("of_wheel_reference", flow_wheel.unsqueeze(0), "ref", True)  # Add batch dimension for logging

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
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
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
            
            # optical flow rgb logs
            if self.use_of_loss:
                of_s_rgb, of_g_rgb = None, None
                if outs["of_s"] is not None:
                    of_s_rgb = flow_to_rgb(outs["of_s"][-1, 0:1])
                if outs["of_g"] is not None:
                    of_g_rgb = flow_to_rgb(outs["of_g"][-1, 0:1])
            else:
                of_s_rgb, of_g_rgb = None, None

            if self.use_pf_loss:
                obs_size = batch["rgb_obs"]["rgb_static"][1:].shape[-1]
                pf_src_s, pf_tgt_s = self.pf_to_rgb(
                    batch["rgb_obs"]["rgb_static"][-2:, 0],
                    batch["p_disp"]["rgb_static"][1:][-1, 0],
                    outs["pf_s"][-1, 0],
                    obs_size,
                )
                if self.use_gripper_camera:
                    pf_src_g, pf_tgt_g = self.pf_to_rgb(
                        batch["rgb_obs"]["rgb_gripper"][-2:, 0],
                        batch["p_disp"]["rgb_gripper"][1:][-1, 0],
                        outs["pf_g"][-1, 0],
                        obs_size,
                    )
                else:
                    pf_src_g, pf_tgt_g = None, None
            else:
                pf_src_s, pf_tgt_s = None, None
                pf_src_g, pf_tgt_g = None, None

            # log gt-pred, optical flow and point flow
            self.log_images(
                robot_id=batch["robot_id"][-1],
                gt_img_s=batch["rgb_obs"]["rgb_static"][-1, 0],
                dcd_img_s=outs["dcd_img_s"][-1, 0],
                pred_img_s=pred_img_s[-1, 0],
                mode="train",
                gt_img_g=gt_img_g_val,
                dcd_img_g=dcd_img_g_val,
                pred_img_g=pred_img_g_val,
                of_s_rgb=of_s_rgb,  
                of_g_rgb=of_g_rgb,
                pf_src_s=pf_src_s,
                pf_tgt_s=pf_tgt_s,
                pf_src_g=pf_src_g,
                pf_tgt_g=pf_tgt_g,
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
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
                # proprio=batch["robot_obs"],
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        for key in losses.keys():
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

        # Keep track of optical flow for logging
        self.val_of_s = outs["of_s"][-1, 0:1] if outs["of_s"] is not None else None
        self.val_of_g = outs["of_g"][-1, 0:1] if outs["of_g"] is not None else None

        # Keep track of point flow for logging
        self.val_pf_s = outs["pf_s"][-1, 0:1] if outs["pf_s"] is not None else None
        self.val_pf_g = outs["pf_g"][-1, 0:1] if outs["pf_g"] is not None else None

        self.val_samples = samples
        self.last_val_batch = batch
        self.last_val_outs = outs
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

        # Log optical flow visualizations
        if self.use_of_loss:
            if self.val_of_s is not None:
                of_s_rgb = flow_to_rgb(self.val_of_s)
            if self.val_of_g is not None:
                of_g_rgb = flow_to_rgb(self.val_of_g)
        else:
            of_s_rgb, of_g_rgb = None, None
        
        if self.use_pf_loss:
            obs_size = self.last_val_batch["rgb_obs"]["rgb_static"][1:].shape[-1]
            pf_src_s, pf_tgt_s = self.pf_to_rgb(
                self.last_val_batch["rgb_obs"]["rgb_static"][-2:, 0],
                self.last_val_batch["p_disp"]["rgb_static"][1:][-1, 0],
                self.last_val_outs["pf_s"][-1, 0],
                obs_size,
            )
            if self.use_gripper_camera:
                pf_src_g, pf_tgt_g = self.pf_to_rgb(
                    self.last_val_batch["rgb_obs"]["rgb_gripper"][-2:, 0],
                    self.last_val_batch["p_disp"]["rgb_gripper"][1:][-1, 0],
                    self.last_val_outs["pf_g"][-1, 0],
                    obs_size,
                )
            else:
                pf_src_g, pf_tgt_g = None, None
        else:
            pf_src_s, pf_tgt_s = None, None
            pf_src_g, pf_tgt_g = None, None

        self.log_images(
            robot_id=self.val_robot_id,
            gt_img_s=self.val_gt_img_s,
            dcd_img_s=self.val_dcd_img_s,
            pred_img_s=pred_img_s[-1, 0],
            mode="val",
            gt_img_g=self.val_gt_img_g,
            dcd_img_g=self.val_dcd_img_g,
            pred_img_g=pred_img_g_val,
            of_s_rgb=of_s_rgb,
            of_g_rgb=of_g_rgb,
            pf_src_s=pf_src_s,
            pf_tgt_s=pf_tgt_s,
            pf_src_g=pf_src_g,
            pf_tgt_g=pf_tgt_g,
        )


    # Optical flow loss functions
    def of_cns_loss(self, rgb: Tensor, flow: Tensor) -> Tensor:
        rgb_1 = rgb[:-1]
        rgb_2 = rgb[1:]

        cns_loss_per_pixel = torch.abs(warp2d(rgb_2, flow) - rgb_1.detach()).sum(dim=[-1, -2, -3])
        return cns_loss_per_pixel

    def of_smoothness_loss(self, rgb: Tensor, flow: Tensor) -> Tensor:
        rgb_1 = rgb[:-1]
        rgb_1, bd = flatten_batch(rgb_1, 3)
        flow, bd = flatten_batch(flow, 3)
        fs_loss = motion_smoothness(flow, rgb_1, order=2)
        fs_loss = unflatten_batch(fs_loss, bd).sum(dim=[-1, -2, -3])
        return fs_loss

    # Point flow loss functions
    def pf_loss(self, gt_p_disp: Tensor, pred_p_disp: Tensor, obs_size: int = 64) -> Tensor:
        return 0.5 * torch.square(pred_p_disp - gt_p_disp * obs_size).sum(dim=[-1, -2])
    
    def pf_to_rgb(self, rgb_obs: Tensor, gt_points_disp: Tensor, pred_points_disp: Tensor, obs_size: int = 64, vis_size: int = 224) -> Tuple[Tensor, Tensor]:
        source_img = resize_img(rgb_obs[0], size=(vis_size, vis_size))
        target_img = resize_img(rgb_obs[1], size=(vis_size, vis_size))

        gt_points_disp_resized = gt_points_disp * vis_size
        pred_points_disp_resized = pred_points_disp * vis_size / obs_size

        source_points_grid = make_source_points_grid(gt_points_disp.shape[0], vis_size)

        source_img_with_points = draw_source_points(source_img, source_points_grid)
        target_img_with_points = draw_target_points(
            target_img, source_points_grid, gt_points_disp_resized, pred_points_disp_resized)
        source_img_with_points = torch.clamp(source_img_with_points, 0, 1)
        target_img_with_points = torch.clamp(target_img_with_points, 0, 1)

        return source_img_with_points, target_img_with_points