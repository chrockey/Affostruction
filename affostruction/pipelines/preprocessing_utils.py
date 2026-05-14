"""
Preprocessing utilities for multiview pipelines.

These functions match the preprocessing used in training datasets to ensure
consistency between training and inference.
"""

from typing import Tuple, Optional
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
    # Find non-zero pixels
    nonzero = np.nonzero(alpha)
    if len(nonzero[0]) == 0:
        # If no valid pixels, return full image bbox
        h, w = alpha.shape
        return (0, 0, w, h)

    # Compute tight bbox
    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()

    # Compute center and size
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    half_size = max(x_max - x_min, y_max - y_min) / 2.0

    # Apply expansion
    expanded_half_size = half_size * size_ratio

    # Apply center offset
    center_x += center_offset[0]
    center_y += center_offset[1]

    # Compute final bbox
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
    # Ensure image has alpha channel
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Extract alpha and compute bbox
    alpha = np.array(image.getchannel(3))
    bbox = compute_bbox_from_alpha(alpha, size_ratio=size_ratio, center_offset=center_offset)

    # Crop to bbox
    image_cropped = image.crop(bbox)

    # Resize to target size
    image_resized = image_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to array and separate channels
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    rgb = image_array[:, :, :3]
    alpha_resized = image_array[:, :, 3]

    # Apply alpha blending
    rgb_blended = rgb * alpha_resized[..., None]

    if return_tensor:
        # Convert to [3, H, W] torch tensor
        return torch.from_numpy(rgb_blended).permute(2, 0, 1).float()
    else:
        return rgb_blended


def preprocess_image_and_depth_with_bbox(
    image: Image.Image,
    depth: np.ndarray,
    target_image_size: int,
    target_depth_size: int | None = None,
    size_ratio: float = 1.2,
    center_offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Preprocess image and depth with synchronized bbox cropping.

    Applies the same bbox crop to both RGB and depth, then resizes to
    potentially different target sizes.

    Args:
        image: PIL Image (must have alpha channel)
        depth: [H, W] depth array (normalized 0-1)
        target_image_size: Target size for RGB image
        target_depth_size: Target size for depth (default: same as image)
        size_ratio: Bbox expansion ratio (default: 1.2)
        center_offset: Bbox center offset (default: (0, 0))

    Returns:
        rgb_tensor: [3, target_image_size, target_image_size] torch.Tensor
        depth_resized: [target_depth_size, target_depth_size] np.ndarray
        alpha_resized: [target_depth_size, target_depth_size] np.ndarray
    """
    if target_depth_size is None:
        target_depth_size = target_image_size

    # Ensure image has alpha channel
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Extract alpha and compute bbox
    alpha = np.array(image.getchannel(3))
    bbox = compute_bbox_from_alpha(alpha, size_ratio=size_ratio, center_offset=center_offset)

    # Crop image to bbox
    image_cropped = image.crop(bbox)

    # Crop depth to same bbox
    left, top, right, bottom = bbox
    h, w = depth.shape
    # Clamp bbox to image bounds
    left_c = max(0, left)
    top_c = max(0, top)
    right_c = min(w, right)
    bottom_c = min(h, bottom)
    depth_cropped = depth[top_c:bottom_c, left_c:right_c]

    # Resize RGB
    image_resized = image_cropped.resize(
        (target_image_size, target_image_size), Image.Resampling.LANCZOS
    )
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    rgb = image_array[:, :, :3]
    alpha_rgb = image_array[:, :, 3]

    # Apply alpha blending to RGB
    rgb_blended = rgb * alpha_rgb[..., None]
    rgb_tensor = torch.from_numpy(rgb_blended).permute(2, 0, 1).float()

    # Resize depth
    depth_pil = Image.fromarray(depth_cropped)
    depth_resized = np.array(
        depth_pil.resize((target_depth_size, target_depth_size), Image.Resampling.LANCZOS)
    ).astype(np.float32)

    # Resize alpha for depth
    alpha_pil = Image.fromarray(alpha_rgb)
    alpha_resized = np.array(
        alpha_pil.resize((target_depth_size, target_depth_size), Image.Resampling.LANCZOS)
    ).astype(np.float32)

    return rgb_tensor, depth_resized, alpha_resized
