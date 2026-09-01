"""
AffordancePipeline: text-conditioned affordance heatmap flow.

Loads an ``ElasticSLatFlowModel`` denoiser + a CLIP text conditioner and
runs CFG flow-matching sampling over sparse coords produced by the
reconstruction pipeline. The output is a per-voxel logit / probability
for each input coord.
"""

import glob
import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .. import models
from ..modules import sparse as sp
from ..modules.text_conditioner import CLIPTextConditioner
from .samplers import FlowEulerCfgSampler


HF_DEFAULT_REPO = "chrockey/Affostruction"
HF_AFFO_SUBFOLDER = "affordance"


def _resolve_hf_artifacts(repo_id: str) -> Tuple[str, str]:
    """Fetch (config.json, model.safetensors) from the affordance subfolder."""
    print(f"Downloading affordance checkpoint from HF: {repo_id}/{HF_AFFO_SUBFOLDER}")
    config_path = hf_hub_download(
        repo_id=repo_id, filename=f"{HF_AFFO_SUBFOLDER}/config.json"
    )
    ckpt_path = hf_hub_download(
        repo_id=repo_id, filename=f"{HF_AFFO_SUBFOLDER}/model.safetensors"
    )
    return config_path, ckpt_path


def _resolve_local_artifacts(src_dir: str) -> Tuple[str, str]:
    """Resolve config + latest ``ckpts/denoiser_ema*.pt`` from a training
    output dir (e.g. ``outputs/stage2_affordance``)."""
    config_path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Affordance config not found: {config_path}")
    candidates = sorted(glob.glob(os.path.join(src_dir, "ckpts", "denoiser_ema*.pt")))
    if not candidates:
        raise FileNotFoundError(
            f"No denoiser_ema*.pt in {os.path.join(src_dir, 'ckpts')}"
        )
    return config_path, candidates[-1]


def _normalize_affo_config(raw: dict) -> dict:
    """Map either the training config or the filtered inference config to a
    flat dict with keys ``denoiser``, ``text_cond_model``, ``sigma_min``,
    ``noise_scale``."""
    if "models" in raw and "trainer" in raw:
        trainer_args = raw["trainer"]["args"]
        return {
            "denoiser": raw["models"]["denoiser"],
            "text_cond_model": trainer_args.get(
                "text_cond_model", "openai/clip-vit-large-patch14"
            ),
            "sigma_min": float(trainer_args.get("sigma_min", 1e-5)),
            "noise_scale": float(trainer_args.get("noise_scale", 5.0)),
        }
    return {
        "denoiser": raw["denoiser"],
        "text_cond_model": raw.get(
            "text_cond_model", "openai/clip-vit-large-patch14"
        ),
        "sigma_min": float(raw.get("sigma_min", 1e-5)),
        "noise_scale": float(raw.get("noise_scale", 5.0)),
    }


def _load_state_dict(ckpt_path: str) -> dict:
    """Load a state dict from either a safetensors file or a torch .pt."""
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors

        return load_safetensors(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


class AffordancePipeline:
    """Text-conditioned affordance heatmap flow.

    Sampling defaults: 50 Euler steps, CFG strength 1.0, initial noise scaled
    by 0.1 and a zero negative condition. The inference noise scale is much
    smaller than the training-time scale (5.0) — the flow is trained on
    logit-space targets, but starting inference from near-zero logits gives
    sharper heatmaps.
    """

    DEFAULT_STEPS = 50
    DEFAULT_CFG_STRENGTH = 1.0
    DEFAULT_NOISE_SCALE = 0.1

    def __init__(
        self,
        denoiser: nn.Module,
        text_conditioner: CLIPTextConditioner,
        sigma_min: float = 1e-5,
        noise_scale: float = DEFAULT_NOISE_SCALE,
    ):
        self.models: dict = {
            "denoiser": denoiser,
            "text_conditioner": text_conditioner,
        }
        self.sigma_min = sigma_min
        self.noise_scale = noise_scale
        self.sampler = FlowEulerCfgSampler(sigma_min=sigma_min)
        self.device = "cpu"
        for m in self.models.values():
            m.eval()

    @staticmethod
    def from_pretrained(source: str = HF_DEFAULT_REPO) -> "AffordancePipeline":
        """Load from either an HF repo id or a local training output dir.

        ``source`` defaults to the public HF repo; passing an existing local
        directory triggers the dev-only local loader instead (see
        ``_resolve_local_artifacts``).
        """
        if os.path.isdir(source):
            config_path, ckpt_path = _resolve_local_artifacts(source)
        else:
            config_path, ckpt_path = _resolve_hf_artifacts(source)

        with open(config_path) as f:
            raw_config = json.load(f)
        config = _normalize_affo_config(raw_config)

        denoiser_name = config["denoiser"]["name"]
        denoiser_args = config["denoiser"]["args"]
        print(f"Building affordance denoiser {denoiser_name}...")
        denoiser = getattr(models, denoiser_name)(**denoiser_args)

        print(f"Loading affordance weights from: {ckpt_path}")
        state = _load_state_dict(ckpt_path)
        denoiser.load_state_dict(state)

        print(f"Loading text conditioner: {config['text_cond_model']}")
        text_conditioner = CLIPTextConditioner(name=config["text_cond_model"])

        return AffordancePipeline(
            denoiser=denoiser,
            text_conditioner=text_conditioner,
            sigma_min=config["sigma_min"],
        )

    def cuda(self) -> "AffordancePipeline":
        self.device = "cuda"
        for k, m in self.models.items():
            self.models[k] = m.cuda()
        return self

    def cpu(self) -> "AffordancePipeline":
        self.device = "cpu"
        for k, m in self.models.items():
            self.models[k] = m.cpu()
        return self

    @torch.no_grad()
    def run(
        self,
        coords: torch.Tensor,
        query: str,
        *,
        steps: int = DEFAULT_STEPS,
        cfg_strength: float = DEFAULT_CFG_STRENGTH,
        noise_scale: Optional[float] = None,
        neg_cond_mode: str = "zeros",
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Args:
            coords: (N, 4) int tensor ``[batch, x, y, z]`` from the
                reconstruction's sparse structure decoder. Single-batch only;
                ``coords[:, 0]`` must all equal 0.
            query: text query (e.g. ``"grasp"``).
            steps: flow-matching Euler steps.
            cfg_strength: classifier-free guidance strength.
            noise_scale: initial-noise scale (default ``DEFAULT_NOISE_SCALE``).
            neg_cond_mode: ``"zeros"`` or ``"empty_text"`` (CLIP embedding of
                the empty string, i.e. the trainer's unconditional embedding).
            seed: optional torch seed.

        Returns:
            dict with:
            - ``coords``: (N, 4) input coords (echoed back)
            - ``logits``: (N,) flow-matching denoised logits
            - ``probs`` : (N,) sigmoid(logits)
            - ``query`` : the input query
        """
        if not isinstance(query, str):
            raise TypeError("AffordancePipeline.run expects a single string query")
        if coords.numel() == 0:
            raise ValueError("coords is empty — reconstruction returned no voxels")
        if int(coords[:, 0].max().item()) != 0 or int(coords[:, 0].min().item()) != 0:
            raise ValueError(
                "AffordancePipeline.run only supports single-batch coords "
                "(coords[:, 0] must be all zero). Use run_batch for many samples."
            )

        out = self.run_batch(
            [coords[:, 1:]],
            [query],
            steps=steps,
            cfg_strength=cfg_strength,
            noise_scale=noise_scale,
            neg_cond_mode=neg_cond_mode,
            seed=seed,
            verbose=verbose,
        )[0]
        out["coords"] = coords.to(self.device).int()
        return out

    @torch.no_grad()
    def run_batch(
        self,
        coords_list: List[torch.Tensor],
        queries: List[str],
        *,
        steps: int = DEFAULT_STEPS,
        cfg_strength: float = DEFAULT_CFG_STRENGTH,
        noise_scale: Optional[float] = None,
        neg_cond_mode: str = "zeros",
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> List[dict]:
        """Sample heatmaps for a batch of (voxel set, query) pairs.

        Args:
            coords_list: list of (N_i, 3) int tensors of voxel indices.
            queries: one text query per entry in ``coords_list``.

        Returns:
            list of dicts with ``logits`` (N_i,), ``probs`` (N_i,), ``query``.
        """
        assert len(coords_list) == len(queries), "one query per coordinate set"
        denoiser = self.models["denoiser"]
        text_cond = self.models["text_conditioner"]

        sigma = self.noise_scale if noise_scale is None else float(noise_scale)
        if seed is not None:
            torch.manual_seed(seed)

        batched = []
        for i, coords in enumerate(coords_list):
            coords = coords.to(self.device).int()
            index = torch.full((coords.shape[0], 1), i, dtype=torch.int32, device=self.device)
            batched.append(torch.cat([index, coords], dim=1))
        batched = torch.cat(batched, dim=0)

        feats = torch.randn(batched.shape[0], denoiser.in_channels, device=self.device) * sigma
        noise = sp.SparseTensor(feats=feats, coords=batched)

        cond = text_cond.encode(list(queries))
        if neg_cond_mode == "zeros":
            neg_cond = torch.zeros_like(cond)
        elif neg_cond_mode == "empty_text":
            neg_cond = text_cond.null_cond().expand_as(cond)
        else:
            raise ValueError(f"Unknown neg_cond_mode: {neg_cond_mode}")

        samples = self.sampler.sample(
            denoiser,
            noise=noise,
            cond=cond,
            neg_cond=neg_cond,
            steps=steps,
            cfg_strength=cfg_strength,
            verbose=verbose,
        ).samples

        outputs = []
        for i, query in enumerate(queries):
            logits = samples.feats[samples.layout[i]].squeeze(-1).float()
            outputs.append(
                {"logits": logits, "probs": torch.sigmoid(logits), "query": query}
            )
        return outputs
