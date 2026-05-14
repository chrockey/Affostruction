"""
Affostruction: Multi-view RGBD to 3D with affordance grounding.

This package provides pipelines for:
1. 3D reconstruction from multi-view RGBD images
2. Text-conditioned affordance heatmap grounding
3. Active view selection based on affordance visibility

Example:
    >>> from affostruction import AffostructionPipeline
    >>> pipeline = AffostructionPipeline.from_pretrained().cuda()
    >>> outputs = pipeline.run(input_dict, queries=["grasp"])
    >>> probs = outputs["affordance"][0]["probs"]   # (N,) per-voxel heatmap
"""

from .affostruction import AffostructionPipeline
from .pipelines.affordance import AffordancePipeline
from .pipelines.reconstruction import ReconstructionPipeline
from .pipelines.view_selection import ViewSelectionPipeline

__all__ = [
    "AffostructionPipeline",
    "ReconstructionPipeline",
    "AffordancePipeline",
    "ViewSelectionPipeline",
]
