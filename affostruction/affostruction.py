"""
AffostructionPipeline: Multi-view RGBD to 3D with affordance grounding.

Orchestrates:
1. ``ReconstructionPipeline`` — multi-view RGBD → sparse coords + SLAT → mesh/gaussian
2. ``AffordancePipeline``     — sparse coords + text query → per-voxel heatmap
3. ``ViewSelectionPipeline``  — affordance-driven next-best view selection
"""

from typing import List, Optional

import torch

from .pipelines.affordance import AffordancePipeline, HF_DEFAULT_REPO
from .pipelines.reconstruction import ReconstructionPipeline
from .pipelines.view_selection import ViewSelectionPipeline


class AffostructionPipeline:
    """
    Affostruction: Multi-view RGBD to 3D with affordance grounding.

    Stages:
    1. ``reconstruction_pipeline``: multi-view RGBD → mesh + sparse coords + SLAT
    2. ``affordance_pipeline``    : sparse coords + text query → per-voxel heatmap
    3. ``view_selection_pipeline``: sparse coords + heatmap → next-best pose

    Example:
        >>> pipeline = AffostructionPipeline.from_pretrained().cuda()
        >>> outputs = pipeline.run(input_dict, queries=["grasp"])
        >>> mesh   = outputs["mesh"][0]
        >>> probs  = outputs["affordance"][0]["probs"]   # (N,) per voxel
    """

    def __init__(
        self,
        reconstruction_pipeline: ReconstructionPipeline,
        affordance_pipeline: Optional["AffordancePipeline"] = None,
        view_selection_pipeline: Optional["ViewSelectionPipeline"] = None,
    ):
        self.reconstruction_pipeline = reconstruction_pipeline
        self.affordance_pipeline = affordance_pipeline
        self.view_selection_pipeline = view_selection_pipeline

    @staticmethod
    def from_pretrained(
        repo_id: str = HF_DEFAULT_REPO,
        load_affordance: bool = True,
        load_view_selection: bool = True,
        affordance_source: Optional[str] = None,
    ) -> "AffostructionPipeline":
        """
        Args:
            repo_id: HF repo id for the reconstruction checkpoint.
            load_affordance: If True, also load the affordance pipeline.
            load_view_selection: If True, attach a ``ViewSelectionPipeline``
                (zero-weight; just hosts hemisphere camera generation + score
                kernels). Disable to skip the auxiliary instantiation.
            affordance_source: Optional override for the affordance checkpoint.
                Accepts either an HF repo id or a local training output dir
                (e.g. ``outputs/heatmap_flow-focal_txt_dit_B_64l8p2_fp16_1m``).
                Defaults to ``repo_id`` (the ``affordance/`` subfolder there).
        """
        reconstruction_pipeline = ReconstructionPipeline.from_pretrained(repo_id)
        affordance_pipeline = None
        if load_affordance:
            affordance_pipeline = AffordancePipeline.from_pretrained(
                affordance_source if affordance_source is not None else repo_id
            )
        view_selection_pipeline = (
            ViewSelectionPipeline() if load_view_selection else None
        )
        return AffostructionPipeline(
            reconstruction_pipeline=reconstruction_pipeline,
            affordance_pipeline=affordance_pipeline,
            view_selection_pipeline=view_selection_pipeline,
        )

    def cuda(self) -> "AffostructionPipeline":
        self.reconstruction_pipeline.cuda()
        if self.affordance_pipeline is not None:
            self.affordance_pipeline.cuda()
        if self.view_selection_pipeline is not None:
            self.view_selection_pipeline.cuda()
        return self

    def cpu(self) -> "AffostructionPipeline":
        self.reconstruction_pipeline.cpu()
        if self.affordance_pipeline is not None:
            self.affordance_pipeline.cpu()
        if self.view_selection_pipeline is not None:
            self.view_selection_pipeline.cpu()
        return self

    def reconstruct(
        self,
        input_dict: dict,
        seed: int = 1,
        sparse_structure_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
        formats: Optional[list] = None,
        return_intermediates: bool = False,
    ) -> dict:
        # ``formats`` is opt-in. Default skips mesh/gaussian decoding so that
        # the common path (affordance heatmap, voxel inspection) does not pay
        # for kaolin/nvdiffrast. Pass ``formats=["mesh", "gaussian"]`` (or a
        # subset) when you need renderable outputs.
        return self.reconstruction_pipeline.run(
            input_dict,
            seed=seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params or {},
            slat_sampler_params=slat_sampler_params or {},
            formats=formats,
            return_intermediates=return_intermediates,
        )

    def predict_affordance(
        self,
        coords: torch.Tensor,
        queries: List[str],
        **kwargs,
    ) -> List[dict]:
        """Run the affordance pipeline given reconstruction coords + queries."""
        if self.affordance_pipeline is None:
            raise RuntimeError(
                "AffordancePipeline not loaded. Pass load_affordance=True to "
                "AffostructionPipeline.from_pretrained."
            )
        return [
            self.affordance_pipeline.run(coords, q, **kwargs) for q in queries
        ]

    def select_view(
        self,
        coords: torch.Tensor,
        probs: torch.Tensor,
        *,
        mode: str = "voxel",
        mesh=None,
        voxel_resolution: Optional[int] = None,
        transforms_path: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Pick the best next-view via affordance visibility.

        ``mode='mesh'`` matches the paper: decode the mesh, paint vertex
        colors with the affordance heatmap, render K hemisphere views, score
        by Σ pixel intensities. ``mode='voxel'`` skips mesh decoding and
        rasterizes voxel centers directly (faster, no nvdiffrast).
        """
        if self.view_selection_pipeline is None:
            raise RuntimeError(
                "ViewSelectionPipeline not loaded. Pass load_view_selection=True "
                "to AffostructionPipeline.from_pretrained."
            )
        if voxel_resolution is None:
            voxel_resolution = int(
                self.reconstruction_pipeline.models["slat_flow_model"].resolution
            )
        return self.view_selection_pipeline.run(
            coords,
            probs,
            voxel_resolution,
            mode=mode,
            mesh=mesh,
            transforms_path=transforms_path,
            **kwargs,
        )

    def run(
        self,
        input_dict: dict,
        queries: Optional[List[str]] = None,
        seed: int = 1,
        sparse_structure_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
        formats: Optional[list] = None,
        affordance_sampler_params: Optional[dict] = None,
        view_selection_mode: Optional[str] = "voxel",
        view_selection_transforms_path: Optional[str] = None,
    ) -> dict:
        """Run reconstruction (and affordance + view selection if requested).

        Args:
            queries: list of affordance queries. Enables the affordance stage.
            view_selection_mode: rendering backend for the active-view
                stage. ``"voxel"`` (default) = per-pixel raycast through
                the dense occupancy/probability volume, decoder-free.
                ``"mesh"`` (opt-in) = paper version: SLAT mesh decode +
                nvdiffrast rasterization (``formats`` must include
                ``"mesh"``). ``None`` skips view selection entirely.
            view_selection_transforms_path: path to a dataset
                ``transforms.json``. When set, candidate poses come from
                those Blender c2w transforms (converted to OpenCV w2c
                per the main-branch convention). Required when view
                selection runs.
        """
        need_intermediates = bool(queries)
        outputs = self.reconstruct(
            input_dict,
            seed=seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            slat_sampler_params=slat_sampler_params,
            formats=formats,
            return_intermediates=need_intermediates,
        )
        if queries:
            outputs["affordance"] = self.predict_affordance(
                outputs["coords"], queries, **(affordance_sampler_params or {})
            )
            if view_selection_mode is not None:
                mesh = outputs.get("mesh", [None])[0] if "mesh" in outputs else None
                outputs["view_selection"] = [
                    self.select_view(
                        outputs["coords"],
                        aff["probs"],
                        mode=view_selection_mode,
                        mesh=mesh,
                        transforms_path=view_selection_transforms_path,
                    )
                    for aff in outputs["affordance"]
                ]
        return outputs


__all__ = [
    "AffostructionPipeline",
    "AffordancePipeline",
    "ReconstructionPipeline",
    "ViewSelectionPipeline",
]
