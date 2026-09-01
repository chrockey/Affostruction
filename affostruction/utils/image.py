"""
Preprocessing utilities for multiview pipelines.

These functions match the preprocessing used in training datasets to ensure
consistency between training and inference.
"""

import numpy as np
import torch
from PIL import Image


def compute_bbox_from_alpha(
    alpha: np.ndarray, size_ratio: float = 1.2, center_offset: tuple[float, float] = (0.0, 0.0)
) -> tuple[int, int, int, int]:
    """
    Compute bounding box from alpha channel.

    Matches the bbox computation used in training datasets:
    - Find tight bbox around non-zero alpha pixels
    - Expand by size_ratio from center
    - Apply optional center offset

    Args:
        alpha: [H, W] alpha channel (0-255 or 0-1)
        size_ratio: Expansion ratio from tight bbox (default: 1.2)
        center_offset: (x, y) offset from bbox center (default: (0, 0))

    Returns:
        (left, top, right, bottom) bbox in image coordinates
    """
    nonzero = np.nonzero(alpha)
    if len(nonzero[0]) == 0:
        h, w = alpha.shape
        return (0, 0, w, h)

    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    half_size = max(x_max - x_min, y_max - y_min) / 2.0

    expanded_half_size = half_size * size_ratio

    center_x += center_offset[0]
    center_y += center_offset[1]

    left = int(center_x - expanded_half_size)
    top = int(center_y - expanded_half_size)
    right = int(center_x + expanded_half_size)
    bottom = int(center_y + expanded_half_size)

    return (left, top, right, bottom)


def preprocess_image_with_bbox(
    image: Image.Image,
    target_size: int = 518,
    size_ratio: float = 1.2,
    center_offset: tuple[float, float] = (0.0, 0.0),
    return_tensor: bool = True,
) -> torch.Tensor | np.ndarray:
    """
    Preprocess image with bbox cropping and alpha blending.

    This matches the preprocessing used in training datasets:
    1. Extract alpha channel
    2. Compute bbox from alpha (with size_ratio expansion)
    3. Crop image to bbox
    4. Resize to target_size
    5. Apply alpha blending to RGB

    Args:
        image: PIL Image (must have alpha channel)
        target_size: Target resolution (default: 518 for pretrained SLAT)
        size_ratio: Bbox expansion ratio (default: 1.2)
        center_offset: Bbox center offset (default: (0, 0))
        return_tensor: Return torch.Tensor if True, else np.ndarray

    Returns:
        If return_tensor=True: [3, H, W] torch.Tensor (float32, 0-1 range)
        If return_tensor=False: [H, W, 3] np.ndarray (float32, 0-1 range)
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = np.array(image.getchannel(3))
    bbox = compute_bbox_from_alpha(alpha, size_ratio=size_ratio, center_offset=center_offset)

    image_cropped = image.crop(bbox)

    image_resized = image_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

    image_array = np.array(image_resized).astype(np.float32) / 255.0
    rgb = image_array[:, :, :3]
    alpha_resized = image_array[:, :, 3]

    rgb_blended = rgb * alpha_resized[..., None]

    if return_tensor:
        return torch.from_numpy(rgb_blended).permute(2, 0, 1).float()
    else:
        return rgb_blended
