"""
3D reconstruction metrics.

* ``volumetric_iou`` — set IoU between predicted and ground-truth occupancy at
  the sparse-structure resolution (64^3).
* ``chamfer_distance_and_fscore`` — Chamfer distance and F-score between 100k
  point clouds sampled from the predicted and ground-truth meshes.
* ``psnr`` / ``LPIPS`` — appearance metrics between renders of the prediction and
  of the ground truth from the same camera (RGB and normal maps).
"""
from typing import Optional, Tuple

import numpy as np
import torch

__all__ = [
    "volumetric_iou",
    "chamfer_distance_and_fscore",
    "nearest_neighbour_distances",
    "psnr",
    "LPIPS",
]


def volumetric_iou(
    pred_coords: np.ndarray, gt_coords: np.ndarray, resolution: int = 64
) -> float:
    """IoU between two sets of integer voxel coordinates on a shared grid.

    Args:
        pred_coords: (N, 3) predicted voxel indices in [0, resolution).
        gt_coords: (M, 3) ground-truth voxel indices in [0, resolution).
        resolution: Grid resolution (kept for signature symmetry / validation).
    """
    pred_set = set(map(tuple, np.asarray(pred_coords, dtype=np.int32)))
    gt_set = set(map(tuple, np.asarray(gt_coords, dtype=np.int32)))
    union = len(pred_set | gt_set)
    return len(pred_set & gt_set) / union if union > 0 else 0.0


@torch.no_grad()
def nearest_neighbour_distances(
    src: np.ndarray, dst: np.ndarray, device: Optional[str] = None, chunk: int = 16384
) -> torch.Tensor:
    """min_j ||src_i - dst_j|| for every i, chunked to bound memory."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src_t = torch.from_numpy(np.asarray(src, dtype=np.float32)).to(device)
    dst_t = torch.from_numpy(np.asarray(dst, dtype=np.float32)).to(device)
    dists = []
    for i in range(0, src_t.shape[0], chunk):
        dists.append(torch.cdist(src_t[i : i + chunk], dst_t).min(dim=1).values)
    return torch.cat(dists)


@torch.no_grad()
def chamfer_distance_and_fscore(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.05,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Chamfer distance (sum of both directional means) and F-score at `threshold`."""
    d_pred_gt = nearest_neighbour_distances(pred_points, gt_points, device).sqrt()
    d_gt_pred = nearest_neighbour_distances(gt_points, pred_points, device).sqrt()
    chamfer = d_pred_gt.mean().item() + d_gt_pred.mean().item()
    precision = (d_pred_gt < threshold).float().mean().item()
    recall = (d_gt_pred < threshold).float().mean().item()
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return chamfer, fscore


def psnr(image_a: np.ndarray, image_b: np.ndarray, max_value: float = 255.0) -> float:
    """Peak signal-to-noise ratio between two uint8 images."""
    mse = np.mean((image_a.astype(np.float64) - image_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(max_value / np.sqrt(mse)))


class LPIPS:
    """Learned perceptual image patch similarity (AlexNet backbone)."""

    def __init__(self, device: Optional[str] = None):
        import lpips as lpips_lib

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips_lib.LPIPS(net="alex", verbose=False).to(self.device)

    @torch.no_grad()
    def __call__(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        def to_tensor(image):
            tensor = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)
            return tensor.permute(2, 0, 1)[None].to(self.device)

        return float(self.model(to_tensor(image_a), to_tensor(image_b)).item())
