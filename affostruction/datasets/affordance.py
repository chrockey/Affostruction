import os
from typing import *
import numpy as np
import torch
from .base import DatasetBase

from ..modules.sparse.basic import SparseTensor
from ..utils.data import load_balanced_group_indices


class AffordanceDataset(DatasetBase):
    """
    Text-conditioned Affordance Flow Matching Dataset

    Loads affordance heatmaps and converts them to logits for flow matching training.

    This dataset:
    - Loads voxel coordinates and affordance heatmaps with text queries
    - Converts affordance probabilities [0, 1] to logits for flow matching
    - Returns both logits (for flow matching) and probabilities (for mask loss)

    Args:
        roots (str): Path to the dataset
        split (Optional[str]): Dataset split ('train', 'val', 'test')
    """

    def __init__(
        self,
        roots: str,
        *,
        split: Optional[Literal["train", "val", "test"]] = None,
        **kwargs,
    ):

        super().__init__(roots, split=split, **kwargs)

        self.loads = [self.metadata.loc[sha256, "num_voxels"] for _, sha256 in self.instances]

    def filter_metadata(self, metadata, root):
        """Filter metadata to only include samples with affordance data."""
        stats = {}

        has_affordance = []
        for idx, row in metadata.iterrows():
            sha256 = row["sha256"]

            affordance_path = os.path.join(root, "affordances", sha256, "affordance.npz")
            has_affordance.append(os.path.exists(affordance_path))

        metadata = metadata[has_affordance]
        stats["With affordance"] = len(metadata)

        return metadata, stats

    def get_instance(self, root, instance):
        """
        Load a single instance with voxel coords and affordance heatmap.

        Returns:
            pack: Dictionary containing:
                - coords: Voxel coordinates [N, 3] at resolution 64
                - feats: Affordance logits [N] (for flow matching)
                - gt_prob: Affordance probabilities [N] (for mask loss)
                - query: Single text query string
        """
        sha256 = instance

        affordance_path = os.path.join(root, "affordances", sha256, "affordance.npz")
        affordance_data = np.load(affordance_path, allow_pickle=True)

        voxel_indices = affordance_data["coords"].astype(np.int64)
        queries = affordance_data["query"]
        heatmaps = affordance_data["heatmap"]

        if self.split == "train":
            query_idx = np.random.randint(0, len(queries))
        else:
            query_idx = 0

        selected_query = str(queries[query_idx])
        selected_heatmap = heatmaps[:, query_idx]

        gt_prob = torch.from_numpy(selected_heatmap).float()

        gt_logit = torch.logit(torch.clamp(gt_prob, 1e-6, 1 - 1e-6))

        pack = {
            "coords": torch.from_numpy(voxel_indices).int(),
            "feats": gt_logit.unsqueeze(-1),
            "gt_prob": gt_prob,
            "query": selected_query,
        }

        return pack

    @staticmethod
    def collate_fn(batch, split_size=None):
        """
        Custom collate function for affordance flow matching with sparse tensor support.

        Args:
            batch: List of samples from __getitem__
            split_size: Optional batch split size for gradient accumulation

        Returns:
            Dictionary with:
                - x_0: Sparse tensor with affordance logits for flow matching
                - cond: List of text queries [B]
                - gt_prob: List of affordance probabilities [B], each [N_i]
        """
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b["coords"].shape[0] for b in batch], split_size
            )

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            coords = []
            feats = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(
                    torch.cat(
                        [torch.full((b["coords"].shape[0], 1), i, dtype=torch.int32), b["coords"]],
                        dim=1,
                    )
                )
                feats.append(b["feats"])
                end = start + b["coords"].shape[0]
                layout.append(range(start, end))
                start = end

            coords = torch.cat(coords, dim=0)
            feats = torch.cat(feats, dim=0)
            pack["x_0"] = SparseTensor(
                feats=feats,
                coords=coords,
                layout=layout,
            )

            pack["cond"] = [b["query"] for b in sub_batch]

            pack["gt_prob"] = [b["gt_prob"] for b in sub_batch]

            packs.append(pack)

        return packs if len(packs) > 1 else packs[0]
