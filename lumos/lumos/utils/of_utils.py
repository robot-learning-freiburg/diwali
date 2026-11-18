# Copy flow-related methods from Flower

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def create_flow_wheel(device: torch.device, wheel_size: int = 256, max_magnitude: float = 10.0) -> Tensor:
    """
    Create an optical flow wheel for visualization reference.

    Args:
        wheel_size: Size of the wheel image (wheel_size x wheel_size)
        max_magnitude: Maximum flow magnitude at the edge of the wheel

    Returns:
        RGB tensor of shape (3, wheel_size, wheel_size) with values in [0, 1]
    """

    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, wheel_size, device=device),
        torch.linspace(-1, 1, wheel_size, device=device),
        indexing="ij",
    )

    # Calculate radius and angle for each pixel
    radius = torch.sqrt(x**2 + y**2)
    angle = torch.atan2(y, x)

    # Create mask for pixels inside the unit circle
    mask = radius <= 1.0

    # Convert polar coordinates to flow vectors
    # Flow magnitude scales linearly with radius
    flow_magnitude = radius * max_magnitude

    # Flow components (u, v)
    flow_u = flow_magnitude * torch.cos(angle)
    flow_v = flow_magnitude * torch.sin(angle)

    # Create flow tensor (2, H, W)
    flow = torch.stack([flow_u, flow_v], dim=0).unsqueeze(0)  # Add batch dimension

    # Convert flow to RGB using existing method
    flow_rgb = flow_to_rgb(flow, max_magnitude=max_magnitude)
    flow_rgb = flow_rgb.squeeze(0)  # Remove batch dimension

    # Apply circular mask (set outside pixels to white)
    mask_3ch = mask.unsqueeze(0).expand(3, -1, -1)
    flow_rgb = torch.where(mask_3ch, flow_rgb, torch.ones_like(flow_rgb))

    return flow_rgb

def flow_to_rgb(flow: Tensor, max_magnitude: Optional[float] = None) -> Tensor:
    """
    Convert optical flow to RGB image for visualization.

    Args:
        flow: Optical flow tensor of shape (..., 2, H, W) where the second-to-last dim contains [u, v]
        max_magnitude: Maximum flow magnitude for normalization. If None, uses the max in the batch.

    Returns:
        RGB tensor of shape (..., 3, H, W) with values in [0, 1]
    """
    # Handle different input shapes
    original_shape = flow.shape
    if flow.dim() == 5:  # S x B x 2 x H x W
        S, B, _, H, W = flow.shape
        flow_flat = flow.view(S * B, 2, H, W)
    elif flow.dim() == 4:  # B x 2 x H x W
        B, _, H, W = flow.shape
        flow_flat = flow
        S = None
    else:
        raise ValueError(f"Expected flow tensor with 4 or 5 dimensions, got {flow.dim()}")

    # Extract u and v components
    u = flow_flat[:, 0]  # (S*B, H, W) or (B, H, W)
    v = flow_flat[:, 1]  # (S*B, H, W) or (B, H, W)

    # Compute magnitude and angle
    magnitude = torch.sqrt(u**2 + v**2)
    angle = torch.atan2(v, u)  # Range: [-π, π]

    # Normalize angle to [0, 1] for hue
    hue = (angle + torch.pi) / (2 * torch.pi)  # Range: [0, 1]

    # Normalize magnitude for saturation
    if max_magnitude is None:
        max_magnitude = magnitude.max()
    if max_magnitude > 0:
        saturation = torch.clamp(magnitude / max_magnitude, 0, 1)
    else:
        saturation = torch.zeros_like(magnitude)

    # Set value to 1 for maximum brightness
    value = torch.ones_like(magnitude)

    # Stack HSV channels
    hsv = torch.stack([hue, saturation, value], dim=1)  # (S*B, 3, H, W) or (B, 3, H, W)

    # Convert HSV to RGB
    rgb = _hsv_to_rgb(hsv)

    # Reshape back to original batch structure if needed
    if S is not None:
        rgb = rgb.view(S, B, 3, H, W)

    return rgb

def _hsv_to_rgb(hsv: Tensor) -> Tensor:
    """
    Convert HSV to RGB.

    Args:
        hsv: HSV tensor of shape (B, 3, H, W) with values in [0, 1]

    Returns:
        RGB tensor of shape (B, 3, H, W) with values in [0, 1]
    """
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    c = v * s
    h_prime = h * 6.0
    x = c * (1 - torch.abs(h_prime % 2 - 1))
    m = v - c

    # Initialize RGB channels
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    # Assign values based on hue sector
    mask0 = (h_prime >= 0) & (h_prime < 1)
    mask1 = (h_prime >= 1) & (h_prime < 2)
    mask2 = (h_prime >= 2) & (h_prime < 3)
    mask3 = (h_prime >= 3) & (h_prime < 4)
    mask4 = (h_prime >= 4) & (h_prime < 5)
    mask5 = (h_prime >= 5) & (h_prime < 6)

    r[mask0] = c[mask0]
    g[mask0] = x[mask0]

    r[mask1] = x[mask1]
    g[mask1] = c[mask1]

    g[mask2] = c[mask2]
    b[mask2] = x[mask2]

    g[mask3] = x[mask3]
    b[mask3] = c[mask3]

    r[mask4] = x[mask4]
    b[mask4] = c[mask4]

    r[mask5] = c[mask5]
    b[mask5] = x[mask5]

    # Add the lightness component
    r = r + m
    g = g + m
    b = b + m

    rgb = torch.stack([r, g, b], dim=1)
    return rgb

def gradient_x(img):
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx

def gradient_y(img):
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy

def gradient_x_2nd(img):
    img_l = F.pad(img, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
    img_r = F.pad(img, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    gx = img_l + img_r - 2 * img
    return gx

def gradient_y_2nd(img):
    img_t = F.pad(img, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
    img_b = F.pad(img, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
    gy = img_t + img_b - 2 * img
    return gy

def motion_smoothness(flow, img, order=1, beta=1):
    """edge aware smoothness"""
    # first order smoothness
    if order == 1:
        f_grad_x = gradient_x(flow)
        f_grad_y = gradient_y(flow)
    # second order smoothness
    elif order == 2:
        f_grad_x = gradient_x_2nd(flow)
        f_grad_y = gradient_y_2nd(flow)
    else:
        raise ValueError

    img_grad_x = gradient_x(img)
    img_grad_y = gradient_y(img)
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x), 1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y), 1, keepdim=True) * beta)

    smoothness_x = f_grad_x * weights_x
    smoothness_y = f_grad_y * weights_y

    return smoothness_x.abs() + smoothness_y.abs()

def warp2d(x, flow, return_masks_flow_inside=False):
    # Handle both S x B x C x H x W and B x C x H x W input shapes
    if x.dim() == 5:  # S x B x C x H x W
        S, B, C, H, W = x.size()
        # Reshape to (S*B) x C x H x W for processing
        # view() requires tensor to be contiguous, while reshape() handles non-contiguous tensors
        # by making a new contiguous copy if needed
        x_reshaped = x.reshape(S * B, C, H, W)
        flow_reshaped = flow.reshape(S * B, 2, H, W) if flow.dim() == 5 else flow
        effective_B = S * B
    else:  # B x C x H x W
        B, C, H, W = x.size()
        x_reshaped = x
        flow_reshaped = flow
        effective_B = B
        S = None

    dtype = x.dtype
    device = x.device

    # Inline shape2pxlcoords functionality
    grid_v, grid_u = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=dtype, device=device),
            torch.arange(0.0, W, dtype=dtype, device=device),
        ],
        indexing="ij",
    )

    grid_uv = torch.stack((grid_u, grid_v), dim=0)  # 2 x H x W

    if effective_B != 0:
        grid_uv = grid_uv.unsqueeze(0).repeat(repeats=(effective_B, 1, 1, 1))  # (S*B) x 2 x H x W

    # Inline oflow2pxlcoords functionality
    pxlcoords = grid_uv + flow_reshaped

    # Use torch.nn.functional.grid_sample for warping
    # Convert coordinates to normalized coordinates [-1, 1] for grid_sample
    pxlcoords_norm = pxlcoords.clone()
    pxlcoords_norm[:, 0] = 2.0 * pxlcoords_norm[:, 0] / (W - 1) - 1.0  # u coordinates
    pxlcoords_norm[:, 1] = 2.0 * pxlcoords_norm[:, 1] / (H - 1) - 1.0  # v coordinates

    # grid_sample expects (B, H, W, 2) format
    pxlcoords_norm = pxlcoords_norm.permute(0, 2, 3, 1)  # (S*B) x H x W x 2

    # Perform the warping
    warped = torch.nn.functional.grid_sample(
        x_reshaped, pxlcoords_norm, mode="bilinear", padding_mode="border", align_corners=True
    )

    # Reshape back to original format if input was S x B x C x H x W
    if S is not None:
        warped = warped.view(S, B, C, H, W)

    if return_masks_flow_inside:
        # Create mask for coordinates that are inside the image bounds
        valid_u = (pxlcoords[:, 0] >= 0) & (pxlcoords[:, 0] <= W - 1)
        valid_v = (pxlcoords[:, 1] >= 0) & (pxlcoords[:, 1] <= H - 1)
        mask = valid_u & valid_v

        if S is not None:
            mask = mask.view(S, B, H, W)
        else:
            mask = mask.view(B, H, W)

        return warped, mask

    return warped