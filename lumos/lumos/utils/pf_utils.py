
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from PIL import Image, ImageDraw
import numpy as np

def resize_img(img: Tensor, size: Tuple[int, int]) -> Tensor:
    resized_img = F.interpolate(
        img.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    return resized_img

def make_source_points_grid(num_points: int, size: int) -> Tensor:
    side = int(num_points ** 0.5)
    if side * side != num_points:
        raise ValueError(f"num_points={num_points} is not a perfect square")

    step = size / side
    offset = step / 2.0

    coords = torch.arange(side, dtype=torch.float32) * step + offset
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1) # (N, 2)

    return points

def draw_target_points(
    img: Tensor, source_points: Tensor, gt_disp: Tensor, pred_disp: Tensor, point_mask: Tensor = None,
    point_color: Tuple[int, int, int] = None,
) -> Tensor:
    gt_disp = gt_disp.detach().cpu()
    pred_disp = pred_disp.detach().cpu()
    source_points = source_points.detach().cpu()
    if point_mask is not None:
        point_mask = point_mask.detach().cpu().bool()
        if point_mask.shape != (source_points.shape[0],):
            raise ValueError("point_mask must have one entry per source point")
    gt_points = source_points + gt_disp
    pred_points = source_points + pred_disp

    img = denormalize_img(img)
    pil_img = tensor_chw_to_pil(img)
    draw = ImageDraw.Draw(pil_img)
    rad = 2

    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
    ]

    for i, (p_gt, p_pred) in enumerate(zip(gt_points, pred_points)):
        if point_mask is not None and not point_mask[i]:
            continue
        color = point_color if point_color is not None else palette[i % len(palette)]
        x_gt, y_gt = float(p_gt[0]), float(p_gt[1])
        x_pred, y_pred = float(p_pred[0]), float(p_pred[1])

        draw.line((x_gt, y_gt, x_pred, y_pred), fill=color, width=1)
        draw.ellipse(
            (x_gt - rad, y_gt - rad, x_gt + rad, y_gt + rad),
            outline=color,
            width=1,
        )
        draw.ellipse(
            (x_pred - rad, y_pred - rad, x_pred + rad, y_pred + rad),
            fill=color,
            outline=color,
        )

    return pil_to_tensor_chw(pil_img)

def draw_source_points(img: Tensor, points: Tensor, point_color: Tuple[int, int, int] = None) -> Tensor:
    points = points.detach().cpu()
    img = denormalize_img(img)
    pil_img = tensor_chw_to_pil(img)
    draw = ImageDraw.Draw(pil_img)
    rad = 2

    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
    ]

    for i, p0 in enumerate(points):
        color = point_color if point_color is not None else palette[i % len(palette)]
        x0, y0 = float(p0[0]), float(p0[1])
        draw.ellipse(
            (x0 - rad, y0 - rad, x0 + rad, y0 + rad),
            fill=color,
            outline=color,
        )

    return pil_to_tensor_chw(pil_img)

def denormalize_img(img: Tensor) -> Tensor:
    if img.min() < 0:
        img = img * 0.5 + 0.5
    return img

def tensor_chw_to_pil(img: Tensor) -> Image.Image:
    img = img.detach().float().cpu().clamp(0, 1)
    img_hwc = (img.permute(1, 2, 0) * 255).byte().numpy()
    return Image.fromarray(img_hwc)

def pil_to_tensor_chw(pil_img: Image.Image) -> Tensor:
    arr = np.array(pil_img)
    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    return tensor
