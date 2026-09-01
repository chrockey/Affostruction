import os
from typing import *
import numpy as np
import torch

from .base import (
    DatasetBase,
    MultiViewVoxelConditioningMixin,
)


class SparseStructureLatent(DatasetBase):
    """
    Sparse structure latent dataset

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        split (str, optional): split to use ("train", "val", "test"). If None, uses all data.
    """

    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        manifest: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)

        super().__init__(roots, split=split, manifest=manifest)

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization["std"]).reshape(-1, 1, 1, 1)

    def filter_metadata(self, metadata, root):
        stats = {}

        metadata = metadata[metadata[f"ss_latent_{self.latent_model}"]]
        stats["With sparse structure latents"] = len(metadata)

        if "aesthetic_score" in metadata.columns and metadata["aesthetic_score"].notna().any():
            scored = metadata["aesthetic_score"].notna()
            metadata = metadata[~scored | (metadata["aesthetic_score"] >= self.min_aesthetic_score)]
            stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)

        return metadata, stats

    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root, "ss_latents", self.latent_model, f"{instance}.npz"))
        z = torch.tensor(latent["mean"]).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            "x_0": z,
        }

        return pack

class ReconstructionDataset(
    MultiViewVoxelConditioningMixin, SparseStructureLatent
):
    """
    Multi-view DINOv2 sparse voxel conditioned sparse structure dataset.
    Loads sparse structure latents + multi-view RGB-D data for DINOv2 sparse voxel conditioning.
    Aggregates voxel features from multiple views.
    """

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for multi-view conditioning.
        Each sample has multiple views.
        """
        return {
            "x_0": torch.stack([item["x_0"] for item in batch], dim=0),
            "cond": [
                item["cond"] for item in batch
            ],
        }
