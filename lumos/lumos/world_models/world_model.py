import logging
from typing import Any, Dict, Tuple, Union

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from pyparsing import Optional
import pytorch_lightning as pl
import torch
from torch import Tensor
import wandb

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class WorldModel(pl.LightningModule):
    """
    The lightning module used for training a world model.
    Args:
    """

    def __init__(self, name: str):
        super(WorldModel, self).__init__()
        self.name = name
        self.save_hyperparameters()

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, obs: Tensor, act: Tensor, reset: Tensor, in_state: Tensor) -> Dict[str, Tensor]:
        """
        Main forward pass for at each step.
        Args:
            obs (Tensor): Observation tensor
            act (Tensor): Action tensor
            reset (Tensor): Reset tensor
            in_state (Tensor): State tensor
        Returns:
            outputs (dict):
                - 'prior' (Tensor): Prior state
                - 'post' (Tensor): Posterior state
                - 'post_samples' (Tensor): Posterior samples
                - 'features' (Tensor): Features
                - 'dcd_img' (Tensor): Decoded image
                - 'dcd_s_obs' (Tensor): Decoded state obs
                - 'dprior' (Tensor): Prior state distribution
                - 'dpost' (Tensor): Posterior state distribution
                - 'dprior_dtch' (Tensor): Prior state distribution detached
                - 'dpost_dtch' (Tensor): Posterior state distribution detached
                - 'out_states' (Tensor): Output states
        """

        raise NotImplementedError

    def dream(self, act: Tensor, in_state: Tensor) -> Tuple[Tensor, Tensor]:
        """Dreams the next state and samples from the prior distribution.
        Args:
            act (Tensor): Action tensor
            in_state (Tensor): State tensor
        Returns:
            Tuple of tensors:
                - (Tensor): Hidden state
                - (Tensor): Latent state
        """
        raise NotImplementedError

    def pred_img(self, prior: Tensor, post_samples: Tensor, features: Tensor) -> Tensor:
        """Decodes a sample from parameterized prior
        Args:
            prior (Tensor): Prior state
            post_samples (Tensor): Posterior samples
            features (Tensor): Features
        Returns:
            Tensor: Decoded image
        """
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        """
        Compute and return the training loss.
        Args:
            batch (dict):
                - 'obs' (Tensor): Observation tensor
                - 'act' (Tensor): Action tensor
                - 'reset' (Tensor): Reset tensor
            batch_idx (int): Integer displaying index of this batch.
        Returns:
            loss tensor
        """

        raise NotImplementedError

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        """
        Compute and return the validation loss.
        Args:
            batch (dict):
                - 'obs' (Tensor): Observation tensor
                - 'act' (Tensor): Action tensor
                - 'reset' (Tensor): Reset tensor
            batch_idx (int): Integer displaying index of this batch.
        Returns:
            loss tensor
        """
        raise NotImplementedError

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the loss.
        Args:
            batch (dict):
                - 'obs' (Tensor): Observation tensor
                - 'act' (Tensor): Action tensor
                - 'reset' (Tensor): Reset tensor
            outs (dict):
                - 'prior' (Tensor): Prior state
                - 'post' (Tensor): Posterior state
                - 'post_samples' (Tensor): Posterior samples
                - 'features' (Tensor): Features
                - 'dcd_img' (Tensor): Decoded image
                - 'dcd_s_obs' (Tensor): Decoded state obs
                - 'dprior' (Tensor): Prior state distribution
                - 'dpost' (Tensor): Posterior state distribution
                - 'dprior_dtch' (Tensor): Prior state distribution detached
                - 'dpost_dtch' (Tensor): Posterior state distribution detached
                - 'out_states' (Tensor): Output states
        Returns:
            metrics (dict):
                - 'loss_total' (Tensor): Total loss
                - 'loss_img' (Tensor): Image reconstruction loss
                - 'loss_recon' (Tensor): State Reconstruction loss
                - 'loss_kl' (Tensor): KL divergence loss
                - 'loss_kl-exact' (Tensor): Exact KL divergence loss
                - 'loss_kl-post' (Tensor): Posterior KL divergence loss
                - 'loss_kl-prior' (Tensor): Prior KL divergence loss
                - 'entropy_prior' (Tensor): Prior entropy
                - 'entropy_post' (Tensor): Posterior entropy
        """
        raise NotImplementedError

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @torch.inference_mode()
    def log_metrics(self, metrics: Dict[str, Tensor], mode: str, sync_dist: bool = False):
        """
        logs losses and metrics
        Args:
            metrics (dict): for example 'loss_total' as key (Tensor)
            mode (str): "train" or "val"
        Returns:
            None
        """
        for key, val in metrics.items():
            info = key.split("_")
            self.log(info[0] + "/{}-".format(mode) + info[1], metrics[key].detach(), sync_dist=sync_dist)

    @torch.inference_mode()
    @rank_zero_only
    def log_image(self, name: str, image: Tensor, mode: str = "train", log=False):
        """
        Log images for visualization using PyTorch Lightning's logger.

        Args:
            name: Name/tag for the image
            image: Image tensor of shape (B, C, H, W) with values in [0, 1]
            step: Optional step number for logging
        """
        # Take only the first image from the batch for logging
        if image.dim() == 4:  # B x C x H x W
            img_to_log = image[0]  # C x H x W
        elif image.dim() == 5:  # S x B x C x H x W
            img_to_log = image[0, 0]  # C x H x W
        else:
            img_to_log = image

        img_to_log = img_to_log.permute(1, 2, 0)  # H x W x C
        img_to_log = torch.clamp(img_to_log, 0, 1).to("cpu").detach().numpy()
        img = wandb.Image(img_to_log, caption=name)
        if log and self.logger and "wandb" in self.logger.__class__.__name__.lower():
            self.logger.experiment.log({f"imgs/{mode}": img})
        return img

    @torch.inference_mode()
    @rank_zero_only
    def log_images(
        self,
        robot_id: int,
        gt_img_s: Tensor,
        dcd_img_s: Tensor,
        pred_img_s: Tensor,
        mode: str,
        gt_img_g: Tensor = None,
        dcd_img_g: Tensor = None,
        pred_img_g: Tensor = None,
        **kwargs
    ):
        """
        logs images
        Args:
            gt_obs (Tensor): Original image
            dcd_obs (Tensor): z reconstructed image
            pred_obs (Tensor): z-hat reconstructed image
        Returns:
            None
        """
        if not self.logger:
            return
        if isinstance(self.trainer.datamodule.train_transforms, list):
            transform = self.trainer.datamodule.train_transforms[robot_id]["out_rgb"]
        else:
            transform = self.trainer.datamodule.train_transforms["out_rgb"]
        rgb_static = transform(gt_img_s.to("cpu").detach().numpy())
        img1 = wandb.Image(rgb_static, caption="gt")
        rgb_static = transform(dcd_img_s.to("cpu").detach().numpy())
        img2 = wandb.Image(rgb_static, caption="dcd")
        rgb_static = transform(pred_img_s.to("cpu").detach().numpy())
        img3 = wandb.Image(rgb_static, caption="pred")
        images = [img1, img2, img3]

        # Conditionally log gripper images
        if gt_img_g is not None and dcd_img_g is not None and pred_img_g is not None:
            rgb_gripper_gt = transform(gt_img_g.to("cpu").detach().numpy())
            img4 = wandb.Image(rgb_gripper_gt, caption="gt_g")
            rgb_gripper_dcd = transform(dcd_img_g.to("cpu").detach().numpy())
            img5 = wandb.Image(rgb_gripper_dcd, caption="dcd_g")
            rgb_gripper_pred = transform(pred_img_g.to("cpu").detach().numpy())
            img6 = wandb.Image(rgb_gripper_pred, caption="pred_g")
            images.extend([img4, img5, img6])
        
        for key, img in kwargs.items():
            if img is not None:
                logged_img = self.log_image(key, img, mode=mode)
                images.append(logged_img)

        if "wandb" in self.logger.__class__.__name__.lower():
            # Log images to wandb
            self.logger.experiment.log({f"imgs/{mode}": images})