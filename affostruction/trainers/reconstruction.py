"""
Stage-1 trainer: multi-view RGB-D conditioned sparse-structure flow.

Views are unprojected to voxels by the dataset; this trainer runs DINOv2 on
the RGB crops, samples per-voxel features at the projected UVs, fuses them
across views (mean over duplicate voxels) and feeds the padded token set to the
denoiser as cross-attention conditioning.
"""
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torchvision import transforms

from .flow_matching import FlowMatchingCFGTrainer
from ..utils import distributed


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    """
    3D Positional Encoding for sparse voxels.
    Pre-computes positional encodings for all possible voxel positions.
    """

    def __init__(self, channels, resolution, dtype_override=None):
        """
        :param channels: The feature dimension for positional encoding.
        :param resolution: Voxel grid resolution (e.g., 16 for 16³ grid).
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
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

        self._precompute_pos_embed()

    def _precompute_pos_embed(self):
        """
        Pre-compute positional embeddings for all voxel positions.
        Returns tensor of shape [resolution, resolution, resolution, org_channels]
        """
        x, y, z = self.resolution, self.resolution, self.resolution

        pos_x = torch.arange(x, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, dtype=self.inv_freq.dtype)

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)

        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)

        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            dtype=self.dtype_override if self.dtype_override is not None else torch.float32,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        emb = emb[:, :, :, : self.org_channels]

        self.register_buffer("pos_embed", emb)

    def forward(self, voxel_indices):
        """
        Get positional encodings for given voxel indices.

        Args:
            voxel_indices: [N, 3] tensor of voxel indices (x, y, z)

        Returns:
            pos_emb: [N, org_channels] positional encodings
        """
        x_idx = voxel_indices[:, 0].long()
        y_idx = voxel_indices[:, 1].long()
        z_idx = voxel_indices[:, 2].long()

        return self.pos_embed[x_idx, y_idx, z_idx]


class VoxelConditioningMixin:
    """
    Mixin for DINOv2 sparse voxel conditioning.

    Extracts DINOv2 features from multi-view RGB-D data, creates sparse voxels and
    adds 3D positional embeddings.

    Args:
        dinov2_model: The DINOv2 model name (e.g., 'dinov2_vitl14_reg').
        voxel_resolution: Resolution of the sparse voxel grid (e.g., 16).
        use_amp: Whether to use automatic mixed precision for DINOv2 inference (default: True).
    """

    def __init__(
        self,
        *args,
        dinov2_model: str = "dinov2_vitl14_reg",
        voxel_resolution: int = 16,
        use_amp: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dinov2_model_name = dinov2_model
        self.voxel_resolution = voxel_resolution
        self.use_amp = use_amp
        self.dinov2_model = None
        self.transform = None
        self.pos_encoder = None

        if "vitl" in dinov2_model or "vitg" in dinov2_model:
            self.dinov2_feat_dim = 1024
        elif "vitb" in dinov2_model:
            self.dinov2_feat_dim = 768
        elif "vits" in dinov2_model:
            self.dinov2_feat_dim = 384
        else:
            self.dinov2_feat_dim = 1024

    @staticmethod
    def prepare_for_training(dinov2_model: str, **kwargs):
        """
        Prepare for training by downloading the DINOv2 model.
        """
        if hasattr(
            super(VoxelConditioningMixin, VoxelConditioningMixin),
            "prepare_for_training",
        ):
            super(
                VoxelConditioningMixin, VoxelConditioningMixin
            ).prepare_for_training(**kwargs)
        torch.hub.load("facebookresearch/dinov2", dinov2_model, pretrained=True)

    def _init_dinov2_model(self):
        """
        Initialize the DINOv2 model (lazy initialization for DDP compatibility).
        """
        if self.dinov2_model is not None:
            return

        with distributed.local_master_first():
            dinov2_model = torch.hub.load(
                "facebookresearch/dinov2", self.dinov2_model_name, pretrained=True
            )
        dinov2_model.eval().cuda()


        self.dinov2_model = dinov2_model
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.pos_encoder = PositionalEncoding3D(
            channels=self.dinov2_feat_dim,
            resolution=self.voxel_resolution,
        ).cuda()

    @torch.no_grad()
    def _extract_dinov2_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 features from RGB images.
        Uses automatic mixed precision (FP16) if use_amp=True.

        Args:
            rgb: [B, 3, H, W] RGB images

        Returns:
            features: [B, feat_dim, n_patch, n_patch] feature maps
        """
        if self.dinov2_model is None:
            self._init_dinov2_model()

        rgb = self.transform(rgb).cuda()

        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features_dict = self.dinov2_model(rgb, is_training=True)
        else:
            features_dict = self.dinov2_model(rgb, is_training=True)

        n_patch = rgb.shape[-1] // 14
        patchtokens = features_dict["x_prenorm"][:, self.dinov2_model.num_register_tokens + 1 :]
        patchtokens = patchtokens.permute(0, 2, 1).reshape(
            rgb.shape[0], self.dinov2_feat_dim, n_patch, n_patch
        )

        return patchtokens


class MultiViewVoxelConditioningMixin(VoxelConditioningMixin):
    """
    Mixin for multi-view DINOv2 sparse voxel conditioning.
    Extends VoxelConditioningMixin to handle multiple views by aggregating voxel
    features. Pads to the batch-wise maximum voxel count.

    Args:
        dinov2_model: The DINOv2 model name (e.g., 'dinov2_vitl14_reg').
        voxel_resolution: Resolution of the sparse voxel grid (e.g., 16).
        use_amp: Whether to use automatic mixed precision for DINOv2 inference (default: True).
    """

    def _batch_pad_voxels(
        self, coords_list: List[torch.Tensor], feats_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad voxels to batch-wise max length with positional embeddings.

        Args:
            coords_list: List of B tensors, each [N_i, 3] voxel coordinates
            feats_list: List of B tensors, each [N_i, feat_dim] features

        Returns:
            batch_voxel_features: [B, max_voxels, feat_dim] with positional embeddings
            batch_padding_mask: [B, max_voxels] (True=valid, False=padding)
        """
        if self.dinov2_model is None:
            self._init_dinov2_model()

        batch_size = len(coords_list)
        max_voxels = max(coords.shape[0] for coords in coords_list)
        if max_voxels == 0:
            max_voxels = 1
            device = coords_list[0].device if len(coords_list) > 0 else "cuda"
            batch_voxel_features = torch.zeros(
                batch_size, 1, self.dinov2_feat_dim, dtype=torch.float32, device=device
            )
            batch_padding_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            return batch_voxel_features, batch_padding_mask

        batch_voxel_features = []
        batch_padding_mask = []

        for coords, feats in zip(coords_list, feats_list):
            num_valid_voxels = coords.shape[0]

            if num_valid_voxels == 0:
                voxel_features = torch.zeros(
                    max_voxels, self.dinov2_feat_dim, dtype=torch.float32, device=coords.device
                )
                padding_mask = torch.zeros(max_voxels, dtype=torch.bool, device=coords.device)
                batch_voxel_features.append(voxel_features)
                batch_padding_mask.append(padding_mask)
                continue

            feats = F.layer_norm(feats, feats.shape[-1:])
            pos_emb = self.pos_encoder(coords)
            valid_feats = feats + pos_emb

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

        batch_voxel_features = torch.stack(
            batch_voxel_features, dim=0
        )
        batch_padding_mask = torch.stack(batch_padding_mask, dim=0)

        return batch_voxel_features, batch_padding_mask

    def _aggregate_multiview_voxels(
        self, voxel_coords_list: List[torch.Tensor], features_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently aggregate voxel features from multiple views using torch_scatter.

        Args:
            voxel_coords_list: List of N tensors, each [M_i, 3] voxel indices
            features_list: List of N tensors, each [M_i, 1024] DINOv2 features

        Returns:
            unique_coords: [K, 3] unique voxel coordinates
            averaged_features: [K, 1024] averaged features (using scatter reduce="mean")
        """
        all_coords = torch.cat(voxel_coords_list, dim=0)
        all_features = torch.cat(features_list, dim=0)

        R = self.voxel_resolution
        voxel_hashes = (
            all_coords[:, 0] * R * R + all_coords[:, 1] * R + all_coords[:, 2]
        )

        unique_hashes, inverse_indices = torch.unique(voxel_hashes, return_inverse=True)

        averaged_features = scatter(
            all_features, inverse_indices, dim=0, reduce="mean"
        )

        num_unique = unique_hashes.shape[0]
        unique_coords = torch.zeros(num_unique, 3, dtype=torch.long, device=all_coords.device)
        unique_coords[:, 0] = unique_hashes // (R * R)
        unique_coords[:, 1] = (unique_hashes % (R * R)) // R
        unique_coords[:, 2] = unique_hashes % R

        return unique_coords, averaged_features

    def _process_batch_multiview_sparse_voxels(self, cond_data_batch: List[List[Dict]]):
        """
        Process a batch of multi-view preprocessed voxel data into sparse voxel features.
        Aggregates voxels from multiple views by averaging.

        Args:
            cond_data_batch: List of B items, each is a list of N view dicts with keys:
                - voxel_coords: [M_i, 3] voxel indices
                - uv_coords: [M_i, 2] UV coordinates for feature sampling
                - rgb: [3, H, W] RGB image for DINOv2

        Returns:
            Tuple of (batch_voxel_features, batch_padding_mask) where:
                batch_voxel_features: [B, max_voxels, feat_dim] tensor (dynamic max_voxels per batch)
                batch_padding_mask: [B, max_voxels] boolean mask (True = valid, False = padding)
        """
        if self.dinov2_model is None:
            self._init_dinov2_model()

        batch_aggregated_coords = []
        batch_aggregated_features = []

        for sample_views in cond_data_batch:
            view_voxel_coords = []
            view_features_list = []

            sample_rgb_batch = torch.stack(
                [view_data["rgb"] for view_data in sample_views], dim=0
            ).cuda()

            batch_features = self._extract_dinov2_features(
                sample_rgb_batch
            )

            for view_idx, view_data in enumerate(sample_views):
                voxel_coords = view_data["voxel_coords"].cuda()
                uv_coords = view_data["uv_coords"].cuda()
                features = batch_features[
                    view_idx : view_idx + 1
                ]

                num_voxels = voxel_coords.shape[0]

                if num_voxels > 0:
                    sampled_features = F.grid_sample(
                        features,
                        uv_coords.unsqueeze(0).unsqueeze(1),
                        mode="bilinear",
                        align_corners=False,
                    )

                    sampled_features = sampled_features.squeeze(0).squeeze(1).permute(1, 0)

                    view_voxel_coords.append(voxel_coords)
                    view_features_list.append(sampled_features)

            if len(view_voxel_coords) > 0:
                aggregated_coords, aggregated_features = self._aggregate_multiview_voxels(
                    view_voxel_coords, view_features_list
                )
            else:
                aggregated_coords = torch.zeros(
                    0, 3, dtype=torch.long, device=sample_rgb_batch.device
                )
                aggregated_features = torch.zeros(
                    0, self.dinov2_feat_dim, dtype=torch.float32, device=sample_rgb_batch.device
                )

            batch_aggregated_coords.append(aggregated_coords)
            batch_aggregated_features.append(aggregated_features)

        batch_voxel_features, batch_padding_mask = self._batch_pad_voxels(
            batch_aggregated_coords, batch_aggregated_features
        )

        return batch_voxel_features, batch_padding_mask

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.

        Args:
            cond: List of B items, each is a list of N view dicts

        Returns:
            Processed conditioning tensor [B, num_voxels, feat_dim]
        """
        kwargs.pop("num_views", None)

        cond, padding_mask = self._process_batch_multiview_sparse_voxels(cond)
        kwargs["cond_mask"] = padding_mask
        kwargs["neg_cond"] = torch.zeros_like(cond)
        cond = super(VoxelConditioningMixin, self).get_cond(cond, **kwargs)
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.

        Args:
            cond: List of B items, each is a list of N view dicts

        Returns:
            Processed conditioning tensor [B, num_voxels, feat_dim]
        """
        kwargs.pop("num_views", None)

        cond, padding_mask = self._process_batch_multiview_sparse_voxels(cond)
        kwargs["cond_mask"] = padding_mask
        kwargs["neg_cond"] = torch.zeros_like(cond)
        cond = super(VoxelConditioningMixin, self).get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data (RGB + depth from first view only).

        Args:
            cond: List of B items, each is a list of N view dicts

        Returns:
            Dict with 'rgb' and 'depth' visualization images (on GPU for distributed gather)
        """
        if isinstance(cond, list) and len(cond) > 0:
            if isinstance(cond[0], list):
                first_view = cond[0][0]
            else:
                first_view = cond[0]
        else:
            first_view = cond

        rgb_image = first_view["rgb"].cuda()
        depth_vis = first_view["depth"].cuda()

        return {
            "image": {"value": rgb_image.unsqueeze(0), "type": "image"},
            "depth": {"value": depth_vis.unsqueeze(0), "type": "image"},
        }


class ReconstructionFlowTrainer(
    MultiViewVoxelConditioningMixin, FlowMatchingCFGTrainer
):
    """Multi-view RGB-D conditioned sparse-structure flow-matching trainer."""

    pass
