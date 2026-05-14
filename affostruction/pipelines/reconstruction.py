"""
ReconstructionPipeline: Multi-view RGBD to 3D reconstruction.

This module provides the core reconstruction functionality for Affostruction,
converting multi-view RGBD images with camera parameters into 3D meshes.
"""

from typing import List, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms
from torch_scatter import scatter

from . import samplers
from .base import Pipeline
from ..modules import sparse as sp
from .. import models


def _get_emb(sin_inp):
    """Gets a base embedding for one dimension with sin and cos intertwined"""
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    """
    3D Positional Encoding for sparse voxels.
    Pre-computes positional encodings for all possible voxel positions.
    """

    def __init__(self, channels: int, resolution: int, dtype_override=None):
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1

        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.dtype_override = dtype_override
        self.channels = channels
        self.resolution = resolution

        # Pre-compute positional encodings for all positions
        self._precompute_pos_embed()

    def _precompute_pos_embed(self):
        """Pre-compute positional embeddings for all voxel positions."""
        x, y, z = self.resolution, self.resolution, self.resolution

        pos_x = torch.arange(x, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, dtype=self.inv_freq.dtype)

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)

        emb_x = _get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)  # [x, 1, 1, channels]
        emb_y = _get_emb(sin_inp_y).unsqueeze(1)  # [y, 1, channels]
        emb_z = _get_emb(sin_inp_z)  # [z, channels]

        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            dtype=self.dtype_override if self.dtype_override is not None else torch.float32,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        # Truncate to original channels
        emb = emb[:, :, :, : self.org_channels]

        # Store as buffer
        self.register_buffer("pos_embed", emb)

    def forward(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        """
        Get positional encodings for voxel coordinates.

        Args:
            voxel_coords: (N, 3) voxel indices

        Returns:
            (N, org_channels) positional encodings
        """
        x_idx = voxel_coords[:, 0].long()
        y_idx = voxel_coords[:, 1].long()
        z_idx = voxel_coords[:, 2].long()
        return self.pos_embed[x_idx, y_idx, z_idx]


HF_DEFAULT_REPO = "chrockey/Affostruction"
HF_RECON_SUBFOLDER = "reconstruction"


def _resolve_recon_artifacts(repo_id: str) -> Tuple[str, str]:
    """Fetch (config.json, model.safetensors) from the reconstruction subfolder of an HF repo."""
    print(f"Downloading reconstruction checkpoint from HF: {repo_id}/{HF_RECON_SUBFOLDER}")
    config_path = hf_hub_download(repo_id=repo_id, filename=f"{HF_RECON_SUBFOLDER}/config.json")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=f"{HF_RECON_SUBFOLDER}/model.safetensors")
    return config_path, ckpt_path


class ReconstructionPipeline:
    """
    Multi-view RGBD to 3D reconstruction pipeline.

    This pipeline:
    1. Extracts DINOv2 features from multi-view RGB images
    2. Projects depth to 3D voxel coordinates
    3. Aggregates features across views
    4. Generates sparse structure via flow matching
    5. Generates SLAT (Structured Latent) via flow matching
    6. Decodes to mesh and gaussian representations
    """

    def __init__(self):
        self.models = {}
        self.device = "cpu"

        # Pipeline configuration (defaults, overwritten by from_pretrained)
        self.voxel_resolution = 16
        self.image_size = 224
        self.dinov2_feat_dim = 1024

        # Sampler parameters (defaults from pretrained config)
        self.sparse_structure_sampler_params = {
            "steps": 12,
            "cfg_strength": 7.5,
        }
        self.slat_sampler_params = {
            "steps": 12,
            "cfg_strength": 3.0,
        }

        # Will be initialized in from_pretrained
        self.sparse_structure_sampler = None
        self.slat_sampler = None
        self.slat_normalization = None
        self.pos_encoder = None
        self.image_cond_model_transform = None

    @staticmethod
    def from_pretrained(
        repo_id: str = HF_DEFAULT_REPO,
        pretrained_slat: str = "microsoft/TRELLIS-image-large",
    ) -> "ReconstructionPipeline":
        """
        Load reconstruction pipeline from HuggingFace.

        Args:
            repo_id: HF repo id (default ``"chrockey/Affostruction"``). The
                reconstruction checkpoint lives under the ``reconstruction/``
                subfolder as ``config.json`` + ``model.safetensors``.
            pretrained_slat: HuggingFace model ID for pretrained SLAT flow and decoders

        Returns:
            ReconstructionPipeline instance
        """
        config_path, ckpt_path = _resolve_recon_artifacts(repo_id)

        # Load base TRELLIS pipeline for SLAT and decoders
        print(f"Loading pretrained SLAT and decoders from {pretrained_slat}...")
        base_pipeline = Pipeline.from_pretrained(pretrained_slat)

        with open(config_path) as f:
            ss_config = json.load(f)

        print(f"Loading sparse structure flow from: {ckpt_path}")
        ss_flow_model = getattr(models, ss_config["denoiser"]["name"])(
            **ss_config["denoiser"]["args"]
        )
        ss_flow_model.load_state_dict(load_safetensors(ckpt_path))

        # Create pipeline instance
        pipeline = ReconstructionPipeline()
        pipeline.models = base_pipeline.models
        pipeline.models["sparse_structure_flow_model"] = ss_flow_model

        pipeline.voxel_resolution = ss_config["voxel_resolution"]
        pipeline.image_size = ss_config["image_size"]

        # Set up samplers from base pipeline
        args = base_pipeline._pretrained_args
        pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]

        pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        pipeline.slat_sampler_params = args["slat_sampler"]["params"]
        pipeline.slat_normalization = args["slat_normalization"]

        pipeline._init_dinov2_model(ss_config["dinov2_model"])

        return pipeline

    def _init_dinov2_model(self, name: str):
        """Initialize the DINOv2 model and related components."""
        print(f"Loading DINOv2 model: {name}")
        dinov2_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model

        # Get feature dimension
        if "vitl" in name or "vitg" in name:
            self.dinov2_feat_dim = 1024
        elif "vitb" in name:
            self.dinov2_feat_dim = 768
        elif "vits" in name:
            self.dinov2_feat_dim = 384
        else:
            self.dinov2_feat_dim = 1024

        # Image transform
        self.image_cond_model_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Positional encoder for voxels
        self.pos_encoder = PositionalEncoding3D(
            channels=self.dinov2_feat_dim,
            resolution=self.voxel_resolution,
        )

    def cuda(self):
        """Move pipeline to CUDA."""
        self.device = "cuda"
        for name, model in self.models.items():
            if isinstance(model, nn.Module):
                self.models[name] = model.cuda()
        if self.pos_encoder is not None:
            self.pos_encoder = self.pos_encoder.cuda()
        return self

    def cpu(self):
        """Move pipeline to CPU."""
        self.device = "cpu"
        for name, model in self.models.items():
            if isinstance(model, nn.Module):
                self.models[name] = model.cpu()
        if self.pos_encoder is not None:
            self.pos_encoder = self.pos_encoder.cpu()
        return self

    def _depth_to_voxel_coords(
        self,
        depth_array: np.ndarray,
        alpha_mask: np.ndarray,
        transform_matrix: np.ndarray,
        depth_min: float,
        depth_max: float,
        camera_angle_x: float,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth image to voxel coordinates and UV coordinates.

        Args:
            depth_array: Normalized depth [0, 1]
            alpha_mask: Alpha mask
            transform_matrix: Camera-to-world matrix [4, 4]
            depth_min: Minimum depth value
            depth_max: Maximum depth value
            camera_angle_x: Horizontal FOV in radians
            height: Image height
            width: Image width

        Returns:
            unique_voxel_coords: (N, 3) voxel indices
            unique_uv_coords: (N, 2) UV coordinates normalized to [-1, 1]
        """
        # Convert normalized depth to absolute depth
        absolute_depth = depth_array * (depth_max - depth_min) + depth_min

        # Valid mask. The dataset's alpha channel is typically a pixel or
        # two wider than the rendered object, while the source 16-bit depth
        # PNG fills empty pixels with the sentinel max (= depth_array ≈ 1).
        # Without this guard those rim pixels would unproject to
        # ``depth_max`` distance and seed voxels behind the real surface.
        DEPTH_BG_NORMALIZED = 0.999
        valid_mask = (
            (absolute_depth > 0)
            & (alpha_mask > 0)
            & (depth_array < DEPTH_BG_NORMALIZED)
        )
        if not valid_mask.any():
            return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

        # Camera intrinsics
        focal_length = width / (2.0 * np.tan(camera_angle_x / 2.0))
        cx, cy = width / 2.0, height / 2.0

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Convert to camera coordinates
        x_cam = (x_coords - cx) * absolute_depth / focal_length
        y_cam = (y_coords - cy) * absolute_depth / focal_length
        z_cam = absolute_depth

        # Stack to [H, W, 3]
        xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        xyz_cam_valid = xyz_cam[valid_mask]

        # Transform to world coordinates
        c2w = np.array(transform_matrix)
        c2w_corrected = c2w.copy()
        c2w_corrected[:3, 1:3] *= -1

        xyz_cam_homo = np.concatenate(
            [xyz_cam_valid, np.ones((xyz_cam_valid.shape[0], 1))], axis=1
        )
        xyz_world_homo = xyz_cam_homo @ c2w_corrected.T
        xyz_world = xyz_world_homo[:, :3]

        # Filter points outside [-0.5, 0.5] bounds
        bounds_mask = (xyz_world >= -0.5) & (xyz_world <= 0.5)
        bounds_mask = bounds_mask.all(axis=1)
        xyz_world = xyz_world[bounds_mask]

        if xyz_world.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

        # Convert to voxel indices
        voxel_coords = ((xyz_world + 0.5) * self.voxel_resolution).astype(np.int64)
        voxel_coords = np.clip(voxel_coords, 0, self.voxel_resolution - 1)

        # Get UV coordinates
        valid_indices = np.argwhere(valid_mask)
        uv_coords = valid_indices[:, [1, 0]].astype(np.float32)
        uv_coords = uv_coords[bounds_mask]

        # Normalize to [-1, 1]
        uv_coords[:, 0] = uv_coords[:, 0] / width * 2 - 1
        uv_coords[:, 1] = uv_coords[:, 1] / height * 2 - 1

        # Aggregate overlapping voxels by averaging UV
        R = self.voxel_resolution
        voxel_codes = voxel_coords[:, 0] * R * R + voxel_coords[:, 1] * R + voxel_coords[:, 2]

        unique_codes, inverse_indices = np.unique(voxel_codes, return_inverse=True)
        num_unique = unique_codes.shape[0]
        unique_uv_coords = np.zeros((num_unique, 2), dtype=np.float32)

        for i in range(num_unique):
            mask = inverse_indices == i
            unique_uv_coords[i] = uv_coords[mask].mean(axis=0)

        # Decode unique voxel coordinates
        unique_voxel_coords = np.zeros((num_unique, 3), dtype=np.int64)
        unique_voxel_coords[:, 0] = unique_codes // (R * R)
        unique_voxel_coords[:, 1] = (unique_codes % (R * R)) // R
        unique_voxel_coords[:, 2] = unique_codes % R

        return unique_voxel_coords, unique_uv_coords

    def _depth_to_voxel_coords_metric(
        self,
        depth_metric: np.ndarray,
        alpha_mask: np.ndarray,
        intrinsics: dict,
        transform_matrix: np.ndarray,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Unproject metric depth with a full pinhole intrinsics matrix and
        normalize the resulting point cloud into the TRELLIS [-0.5, 0.5]^3 cube.

        Unlike `_depth_to_voxel_coords`, this path:
          - uses true (fx, fy, cx, cy) so off-center principal points work,
          - consumes metric depth directly (meters), no [0, 1] normalization,
          - derives object scale from the masked point cloud's AABB so any
            real-world object size maps into the canonical cube.

        Args:
            depth_metric: (H, W) float32 metric depth in meters (already resized
                to (height, width))
            alpha_mask: (H, W) object mask; nonzero = foreground
            intrinsics: dict with keys fx, fy, cx, cy already expressed at the
                (height, width) resolution of `depth_metric`
            transform_matrix: (4, 4) canonical camera-to-world. Identity gives a
                front-facing view after the TRELLIS Y/Z flip.

        Returns:
            unique_voxel_coords: (N, 3) int64 voxel indices
            unique_uv_coords: (N, 2) float32 UVs in [-1, 1]
            norm_info: dict with centroid/scale used for normalization (useful
                for downstream debugging or re-projection)
        """
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

        valid_mask = (depth_metric > 0) & (alpha_mask > 0)
        empty = (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
            {"centroid": np.zeros(3, dtype=np.float32), "scale": 1.0},
        )
        if not valid_mask.any():
            return empty

        y_coords, x_coords = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )

        x_cam = (x_coords - cx) * depth_metric / fx
        y_cam = (y_coords - cy) * depth_metric / fy
        z_cam = depth_metric

        xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        xyz_cam_valid = xyz_cam[valid_mask]

        # Normalize into [-0.5, 0.5]^3 with in-plane scale + front-face anchor.
        # Both knobs are derived from the voxel grid rather than hand-tuned:
        #   - one voxel of padding on every side (`margin = 1/voxel_resolution`)
        #   - in-plane AABB scaled to fill the cube up to that margin
        #   - closest-to-camera surface snapped to `y = -0.5 + margin` (front
        #     voxel row), leaving the entire rest of the depth axis as
        #     empty space for the sparse-structure flow to populate with
        #     occluded back geometry.
        # The depth axis is intentionally excluded from scale estimation: a
        # single-view slab is much thinner along depth than the object's
        # true cross-section, so the in-plane extent acts as the proxy for
        # true object size (valid when the view is roughly head-on).
        pc_min = xyz_cam_valid.min(axis=0)
        pc_max = xyz_cam_valid.max(axis=0)
        in_plane_extent = float(max(pc_max[0] - pc_min[0], pc_max[1] - pc_min[1]))
        if in_plane_extent <= 0:
            return empty
        margin = 1.0 / self.voxel_resolution
        scale = (1.0 - 2.0 * margin) / in_plane_extent
        front_anchor = -0.5 + margin
        centroid = np.array(
            [
                (pc_min[0] + pc_max[0]) * 0.5,
                (pc_min[1] + pc_max[1]) * 0.5,
                pc_min[2] - front_anchor / scale,
            ],
            dtype=xyz_cam_valid.dtype,
        )
        xyz_cam_valid = (xyz_cam_valid - centroid) * scale

        # Camera -> TRELLIS world: negate Y,Z rotation columns (matches the
        # convention used by `_depth_to_voxel_coords`). With identity c2w this
        # yields a front-facing view with Y up, Z backward.
        c2w = np.array(transform_matrix, dtype=np.float64)
        c2w_corrected = c2w.copy()
        c2w_corrected[:3, 1:3] *= -1
        xyz_cam_homo = np.concatenate(
            [xyz_cam_valid, np.ones((xyz_cam_valid.shape[0], 1))], axis=1
        )
        xyz_world = (xyz_cam_homo @ c2w_corrected.T)[:, :3]

        bounds_mask = (xyz_world >= -0.5) & (xyz_world <= 0.5)
        bounds_mask = bounds_mask.all(axis=1)
        xyz_world = xyz_world[bounds_mask]
        if xyz_world.shape[0] == 0:
            return empty

        voxel_coords = ((xyz_world + 0.5) * self.voxel_resolution).astype(np.int64)
        voxel_coords = np.clip(voxel_coords, 0, self.voxel_resolution - 1)

        valid_indices = np.argwhere(valid_mask)
        uv_coords = valid_indices[:, [1, 0]].astype(np.float32)
        uv_coords = uv_coords[bounds_mask]
        uv_coords[:, 0] = uv_coords[:, 0] / width * 2 - 1
        uv_coords[:, 1] = uv_coords[:, 1] / height * 2 - 1

        R = self.voxel_resolution
        voxel_codes = voxel_coords[:, 0] * R * R + voxel_coords[:, 1] * R + voxel_coords[:, 2]
        unique_codes, inverse_indices = np.unique(voxel_codes, return_inverse=True)
        num_unique = unique_codes.shape[0]
        unique_uv_coords = np.zeros((num_unique, 2), dtype=np.float32)
        for i in range(num_unique):
            mask = inverse_indices == i
            unique_uv_coords[i] = uv_coords[mask].mean(axis=0)

        unique_voxel_coords = np.zeros((num_unique, 3), dtype=np.int64)
        unique_voxel_coords[:, 0] = unique_codes // (R * R)
        unique_voxel_coords[:, 1] = (unique_codes % (R * R)) // R
        unique_voxel_coords[:, 2] = unique_codes % R

        norm_info = {
            "centroid": centroid.astype(np.float32),
            "scale": float(scale),
        }
        return unique_voxel_coords, unique_uv_coords, norm_info

    @torch.no_grad()
    def _extract_dinov2_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 features from RGB images.

        Args:
            rgb: (B, 3, H, W) RGB images

        Returns:
            features: (B, feat_dim, n_patch, n_patch) feature maps
        """
        rgb = self.image_cond_model_transform(rgb).to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features_dict = self.models["image_cond_model"](rgb, is_training=True)

        n_patch = rgb.shape[-1] // 14  # DINOv2 patch size is 14
        patchtokens = features_dict["x_prenorm"][
            :, self.models["image_cond_model"].num_register_tokens + 1 :
        ]

        patchtokens = patchtokens.permute(0, 2, 1).reshape(
            rgb.shape[0], self.dinov2_feat_dim, n_patch, n_patch
        )

        return patchtokens

    def _aggregate_multiview_voxels(
        self, voxel_coords_list: List[torch.Tensor], features_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate voxel features from multiple views.

        Returns:
            unique_coords: (K, 3) unique voxel coordinates
            averaged_features: (K, feat_dim) averaged features
        """
        all_coords = torch.cat(voxel_coords_list, dim=0)
        all_features = torch.cat(features_list, dim=0)

        # Hash voxel coordinates
        R = self.voxel_resolution
        voxel_hashes = all_coords[:, 0] * R * R + all_coords[:, 1] * R + all_coords[:, 2]

        # Find unique voxels
        unique_hashes, inverse_indices = torch.unique(voxel_hashes, return_inverse=True)

        # Average features
        averaged_features = scatter(all_features, inverse_indices, dim=0, reduce="mean")

        # Decode coordinates
        num_unique = unique_hashes.shape[0]
        unique_coords = torch.zeros(num_unique, 3, dtype=torch.long, device=all_coords.device)
        unique_coords[:, 0] = unique_hashes // (R * R)
        unique_coords[:, 1] = (unique_hashes % (R * R)) // R
        unique_coords[:, 2] = unique_hashes % R

        return unique_coords, averaged_features

    def _batch_pad_voxels(
        self, coords_list: List[torch.Tensor], feats_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad voxels to batch-wise max length with positional embeddings.

        Returns:
            batch_voxel_features: (B, max_voxels, feat_dim)
            batch_padding_mask: (B, max_voxels) (True=valid, False=padding)
        """
        max_voxels = max(coords.shape[0] for coords in coords_list)

        batch_voxel_features = []
        batch_padding_mask = []

        for coords, feats in zip(coords_list, feats_list):
            num_valid_voxels = coords.shape[0]

            # Apply layer norm
            feats = F.layer_norm(feats, feats.shape[-1:])

            # Add positional embeddings
            pos_emb = self.pos_encoder(coords).to(feats.device)
            valid_feats = feats + pos_emb

            # Pad to max_voxels
            num_padding = max_voxels - num_valid_voxels
            if num_padding > 0:
                padding_feats = torch.zeros(
                    num_padding,
                    self.dinov2_feat_dim,
                    dtype=valid_feats.dtype,
                    device=valid_feats.device,
                )
                voxel_features = torch.cat([valid_feats, padding_feats], dim=0)
                padding_mask = torch.cat(
                    [
                        torch.ones(num_valid_voxels, dtype=torch.bool, device=valid_feats.device),
                        torch.zeros(num_padding, dtype=torch.bool, device=valid_feats.device),
                    ],
                    dim=0,
                )
            else:
                voxel_features = valid_feats
                padding_mask = torch.ones(
                    num_valid_voxels, dtype=torch.bool, device=valid_feats.device
                )

            batch_voxel_features.append(voxel_features)
            batch_padding_mask.append(padding_mask)

        batch_voxel_features = torch.stack(batch_voxel_features, dim=0)
        batch_padding_mask = torch.stack(batch_padding_mask, dim=0)

        return batch_voxel_features, batch_padding_mask

    def get_cond(self, input_dict: dict) -> dict:
        """
        Get conditioning from multiview RGBD input.

        Args:
            input_dict: Dictionary with keys:
                - 'images': List[PIL.Image]
                - 'depths': List[np.ndarray]
                - 'alphas': List[np.ndarray]
                - 'camera_params': List[dict]

        Returns:
            dict: Conditioning with keys 'cond', 'neg_cond', 'cond_mask'
        """
        images = input_dict["images"]
        depths = input_dict["depths"]
        alphas = input_dict["alphas"]
        camera_params = input_dict["camera_params"]

        view_voxel_coords = []
        view_features_list = []

        # Prepare RGB images for batched DINOv2 inference
        rgb_tensors = []
        for img in images:
            # Ensure image has alpha channel
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Resize
            img_resized = img.resize((self.image_size, self.image_size), Image.LANCZOS)

            # Convert to array
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]

            # Apply alpha blending
            rgb_blended = rgb * alpha[..., None]

            rgb_tensor = torch.from_numpy(rgb_blended).permute(2, 0, 1).float()
            rgb_tensors.append(rgb_tensor)

        rgb_batch = torch.stack(rgb_tensors, dim=0).to(self.device)

        # Extract DINOv2 features
        batch_features = self._extract_dinov2_features(rgb_batch)

        # Process each view
        for view_idx, (depth, alpha, cam_params) in enumerate(zip(depths, alphas, camera_params)):
            is_metric = bool(cam_params.get("metric_depth", False))

            if is_metric:
                # Metric depth: zero out background before resize, then use
                # NEAREST on both depth and mask so boundary pixels don't
                # interpolate across depth discontinuities (bilinear would
                # bleed background zeros into foreground and inflate the
                # point cloud's z extent, corrupting the scale normalization).
                orig_h, orig_w = depth.shape[:2]
                depth_masked = depth * (alpha > 0).astype(depth.dtype)
                depth_resized = np.array(
                    Image.fromarray(depth_masked).resize(
                        (self.image_size, self.image_size), Image.NEAREST
                    )
                ).astype(np.float32)
                alpha_resized = np.array(
                    Image.fromarray(alpha).resize(
                        (self.image_size, self.image_size), Image.NEAREST
                    )
                ).astype(np.float32)

                # Rescale intrinsics to the resized resolution.
                sx = self.image_size / float(orig_w)
                sy = self.image_size / float(orig_h)
                src_K = cam_params["intrinsics"]
                intrinsics_scaled = {
                    "fx": float(src_K["fx"]) * sx,
                    "fy": float(src_K["fy"]) * sy,
                    "cx": float(src_K["cx"]) * sx,
                    "cy": float(src_K["cy"]) * sy,
                }
                voxel_coords, uv_coords, _ = self._depth_to_voxel_coords_metric(
                    depth_resized,
                    alpha_resized,
                    intrinsics_scaled,
                    cam_params["transform_matrix"],
                    self.image_size,
                    self.image_size,
                )
            else:
                depth_resized = np.array(
                    Image.fromarray(depth).resize(
                        (self.image_size, self.image_size), Image.LANCZOS
                    )
                ).astype(np.float32)
                alpha_resized = np.array(
                    Image.fromarray(alpha).resize(
                        (self.image_size, self.image_size), Image.LANCZOS
                    )
                ).astype(np.float32)

                voxel_coords, uv_coords = self._depth_to_voxel_coords(
                    depth_resized,
                    alpha_resized,
                    cam_params["transform_matrix"],
                    cam_params["depth_min"],
                    cam_params["depth_max"],
                    cam_params["camera_angle_x"],
                    self.image_size,
                    self.image_size,
                )

            if voxel_coords.shape[0] > 0:
                voxel_coords = torch.from_numpy(voxel_coords).long().to(self.device)
                uv_coords = torch.from_numpy(uv_coords).float().to(self.device)

                # Sample features at UV coordinates
                features = batch_features[view_idx : view_idx + 1]
                sampled_features = F.grid_sample(
                    features,
                    uv_coords.unsqueeze(0).unsqueeze(1),
                    mode="bilinear",
                    align_corners=False,
                )
                sampled_features = sampled_features.squeeze(0).squeeze(1).permute(1, 0)

                view_voxel_coords.append(voxel_coords)
                view_features_list.append(sampled_features)

        # Aggregate voxels from all views
        if len(view_voxel_coords) > 0:
            aggregated_coords, aggregated_features = self._aggregate_multiview_voxels(
                view_voxel_coords, view_features_list
            )
        else:
            aggregated_coords = torch.zeros(0, 3, dtype=torch.long, device=self.device)
            aggregated_features = torch.zeros(
                0, self.dinov2_feat_dim, dtype=torch.float32, device=self.device
            )

        # Batch-wise padding
        cond, padding_mask = self._batch_pad_voxels([aggregated_coords], [aggregated_features])

        neg_cond = torch.zeros_like(cond)
        return {
            "cond": cond,
            "neg_cond": neg_cond,
            "cond_mask": padding_mask,
        }

    def get_cond_single_image(self, input_dict: dict) -> dict:
        """
        Get conditioning from single (first) image for pretrained SLAT flow.

        Args:
            input_dict: Dictionary with 'images' key

        Returns:
            dict: Conditioning with keys 'cond', 'neg_cond'
        """
        from .preprocessing_utils import preprocess_image_with_bbox

        images = input_dict["images"]
        img = images[0]

        # Preprocess first image with bbox crop + alpha blending at 518x518
        rgb_tensor = (
            preprocess_image_with_bbox(img, target_size=518, size_ratio=1.2, return_tensor=True)
            .unsqueeze(0)
            .to(self.device)
        )

        # Extract DINOv2 features
        rgb_batch = self.image_cond_model_transform(rgb_tensor).to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = self.models["image_cond_model"](rgb_batch, is_training=True)["x_prenorm"]

        # Apply layer norm
        patchtokens = F.layer_norm(features, features.shape[-1:])

        neg_cond = torch.zeros_like(patchtokens)

        return {
            "cond": patchtokens,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond: Conditioning dictionary
            num_samples: Number of samples to generate
            sampler_params: Additional sampler parameters

        Returns:
            torch.Tensor: Coordinates of the sparse structure (N, 4)
        """
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)

        # Merge default params with overrides
        params = {**self.sparse_structure_sampler_params, **(sampler_params or {})}

        # Apply noise_scale if provided
        noise_scale = params.pop("noise_scale", 1.0)
        noise = noise * noise_scale

        # Extract conditioning arguments
        cond_args = {k: v for k, v in cond.items() if k in ["cond", "neg_cond", "cond_mask"]}

        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond_args, **params, verbose=True
        ).samples

        # Decode occupancy latent
        decoder = self.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = None,
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond: Conditioning dictionary
            coords: Coordinates of the sparse structure
            sampler_params: Additional sampler parameters

        Returns:
            sp.SparseTensor: Sampled structured latent
        """
        flow_model = self.models["slat_flow_model"]
        noise_feats = torch.randn(coords.shape[0], flow_model.in_channels).to(self.device)

        # Merge default params with overrides
        params = {**self.slat_sampler_params, **(sampler_params or {})}

        # Apply noise_scale if provided
        noise_scale = params.pop("noise_scale", 1.0)
        noise_feats = noise_feats * noise_scale

        noise = sp.SparseTensor(
            feats=noise_feats,
            coords=coords,
        )

        # Extract conditioning arguments
        cond_args = {k: v for k, v in cond.items() if k in ["cond", "neg_cond", "cond_mask"]}

        slat = self.slat_sampler.sample(
            flow_model, noise, **cond_args, **params, verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat: Structured latent
            formats: Output formats ("mesh", "gaussian", "radiance_field")

        Returns:
            dict: Decoded outputs
        """
        ret = {}
        if "mesh" in formats:
            ret["mesh"] = self.models["slat_decoder_mesh"](slat)
        if "gaussian" in formats:
            ret["gaussian"] = self.models["slat_decoder_gs"](slat)
        if "radiance_field" in formats:
            ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    @torch.no_grad()
    def run(
        self,
        input_dict: dict,
        num_samples: int = 1,
        seed: int = 1,
        sparse_structure_sampler_params: dict = None,
        slat_sampler_params: dict = None,
        formats: Optional[List[str]] = None,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Run the full reconstruction pipeline.

        Args:
            input_dict: Dictionary with keys:
                - 'images': List[PIL.Image]
                - 'depths': List[np.ndarray]
                - 'alphas': List[np.ndarray]
                - 'camera_params': List[dict]
            num_samples: Number of samples to generate
            seed: Random seed
            sparse_structure_sampler_params: Sparse structure sampler overrides
            slat_sampler_params: SLAT sampler overrides
            formats: Output formats
            return_intermediates: If True, the returned dict also contains
                ``coords`` (sparse coords at the SS-decoder resolution) and
                ``slat`` (the sampled structured latent). Needed for the
                affordance pipeline.

        Returns:
            dict: Generated 3D assets
        """
        # Get conditioning for sparse structure
        cond_ss = self.get_cond(input_dict)

        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(
            cond_ss, num_samples, sparse_structure_sampler_params
        )

        # Get conditioning for SLAT (use single image for pretrained SLAT)
        cond_slat = self.get_cond_single_image(input_dict)

        slat = self.sample_slat(cond_slat, coords, slat_sampler_params)
        # ``formats=None`` skips mesh/gaussian decoding entirely. Decoding is
        # only needed for rendering (textured GLB, gaussian PLY, side-by-side
        # video) — the affordance pipeline runs on coords + slat directly, so
        # we keep decoding opt-in to avoid pulling in nvdiffrast/kaolin when
        # callers only want the heatmap.
        outputs = self.decode_slat(slat, formats if formats is not None else [])

        if return_intermediates:
            outputs["coords"] = coords
            outputs["slat"] = slat

        return outputs
