from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from diwali.env.utils.calvin_helpers import resize_image
from cotracker.utils.visualizer import Visualizer
import time

device = 'cuda'
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

def get_points_on_a_grid(grid_size, img_size, device="cuda"):
    """Same as cotracker.models.core.model_utils.get_points_on_a_grid,
    except the margin scales with grid_size."""
    center = [img_size[0] / 2, img_size[1] / 2]
    margin = img_size[1] / grid_size / 2
    range_y = (margin - img_size[0] / 2 + center[0], img_size[0] / 2 + center[0] - margin)
    range_x = (margin - img_size[1] / 2 + center[1], img_size[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, grid_size, device=device),
        torch.linspace(*range_x, grid_size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def build_grid_queries(grid_size, img_size, device="cuda"):
    grid_pts = get_points_on_a_grid(grid_size, img_size, device=device)
    return torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)


def get_points_displacement(frames, grid_size, img_resolution):
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
    queries = build_grid_queries(grid_size, video.shape[3:], device=device)
    pred_tracks, pred_visibility = cotracker(video, queries=queries)  # B T N 2,  B T N 1

    if pred_tracks.shape[1] > 2:
        pred_tracks = pred_tracks[:, ::8]

    points_disp = pred_tracks[0][1] - pred_tracks[0][0]
    points_disp = points_disp / img_resolution

    return points_disp


def zero_displacement(grid_size):
    return torch.zeros(grid_size * grid_size, 2, device=device, dtype=torch.float32) # B T N 2


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="calculate_points_displacement")
def process_dataset(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    keys = cfg.desired_keys
    step = cfg.action_chunk_size

    img_size = cfg.img_resolution
    if cfg.rgb_interpolation == "Area":
        rgb_interpolation = cv2.INTER_AREA
    elif cfg.rgb_interpolation == "Cubic":
        rgb_interpolation = cv2.INTER_CUBIC
    else:
        raise ValueError(f"Invalid interpolation method: {cfg.rgb_interpolation}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in cfg.splits:
        split_path = Path(input_dir) / split
        output_split_path = Path(output_dir) / split
        output_split_path.mkdir(parents=True, exist_ok=True)

        files = sorted(split_path.glob("episode_*.npz"))
        extracted_imgs = {}
        for key in keys:
            extracted_imgs[key] = {}

        for npz_file in tqdm(files, desc=f"Reading {split} data"):
            ep_id = int(npz_file.stem.split("_")[-1])
            data = np.load(npz_file)
            for key in keys:
                assert key in data, f"Key '{key}' not found in {npz_file}"
                extracted_imgs[key][ep_id] = resize_image(data[key], intp=rgb_interpolation, resolution=img_size)

        ep_start_end_ids = np.load(split_path / "ep_start_end_ids.npy", allow_pickle=True)
        num = 0
        for ep_start, ep_end in tqdm(ep_start_end_ids):
            num += ep_end - ep_start + 1
            for ep_id in range(ep_start, ep_start + step):
                points_disp = {}
                for key in keys:
                    points_disp[key] = zero_displacement(cfg.grid_size)
                output_file = output_split_path / f"episode_{ep_id:0{cfg.n_digits}d}.npz"
                data_npz = {k: v.cpu().numpy() for k, v in points_disp.items()}
                np.savez_compressed(output_file, **data_npz)

            for ep_id in tqdm(range(ep_start + step, ep_end + 1), desc=f"Processing {split} data pairs"):
                prev_ep_id = ep_id - step
                points_disp = {}
                for key in keys:
                    assert key in extracted_imgs, f"Key '{key}' not found in extracted_imgs"
                    assert ep_id in extracted_imgs[key], f"Episode id {ep_id} not found in extracted_imgs[{key}]"
                    assert prev_ep_id in extracted_imgs[key], f"Previous episode id {prev_ep_id} not found in extracted_imgs[{key}]"
                    if cfg.add_intermediate_frames is True:
                        frames = np.stack([extracted_imgs[key][id] for id in range(prev_ep_id, ep_id + 1)])
                    else:
                        frames = np.stack([extracted_imgs[key][id] for id in range(prev_ep_id, ep_id + 1, step)])

                    points_disp[key] = get_points_displacement(frames, cfg.grid_size, img_size)

                    output_file = output_split_path / f"episode_{ep_id:0{cfg.n_digits}d}.npz"
                    data_npz = {k: v.cpu().numpy() for k, v in points_disp.items()}
                    np.savez_compressed(output_file, **data_npz)
        print(f"Processed {num} episodes in split '{split}'")


if __name__ == "__main__":
    process_dataset()