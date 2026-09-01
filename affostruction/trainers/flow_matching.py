"""
Flow-matching trainers.

``FlowMatchingTrainer`` implements the rectified-flow objective on dense
tensors, ``SparseFlowMatchingTrainer`` the sparse-tensor variant used by the
affordance stage, and the ``CFG`` subclasses add classifier-free-guidance
condition dropout.
"""
from typing import *
import copy
import functools

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from .basic import BasicTrainer
from ..modules import sparse as sp
from ..pipelines import samplers
from ..utils.data import BalancedResumableSampler, cycle, recursive_to_device
from ..utils.general import dict_foreach, dict_reduce


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"

        if self.p_uncond > 0:
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)

            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    mask = torch.tensor(mask, device=cond.device).reshape(
                        -1, *[1] * (cond.ndim - 1)
                    )
                    return torch.where(mask, neg_cond, cond)
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            mask = list(np.random.rand(B) < self.p_uncond)
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))

        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {"cond": cond, "neg_cond": neg_cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """

    def __init__(
        self,
        *args,
        t_schedule: dict = {
            "name": "logitNormal",
            "args": {
                "mean": 0.0,
                "std": 1.0,
            },
        },
        sigma_min: float = 1e-5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

    def diffuse(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {"cond": cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)

    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule["name"] == "uniform":
            t = torch.rand(batch_size)
        elif self.t_schedule["name"] == "logitNormal":
            mean = self.t_schedule["args"]["mean"]
            std = self.t_schedule["args"]["std"]
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(self, x_0: torch.Tensor, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        mse_per_instance = np.array(
            [F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])]
        )
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    def _slice_batch(self, data: Any, batch_size: int) -> Any:
        """
        Recursively slice data structures to the given batch size.
        Handles tensors, dicts, lists, and other types.
        """
        if isinstance(data, torch.Tensor):
            return data[:batch_size].cuda()
        elif isinstance(data, dict):
            return {k: self._slice_batch(v, batch_size) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(data[:batch_size])
        else:
            try:
                return data[:batch_size]
            except (TypeError, KeyError):
                return data

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = self._slice_batch(data, batch)
            noise = torch.randn_like(data["x_0"])
            sample_gt.append(data["x_0"])
            cond_vis.append(self.vis_cond(**data))
            del data["x_0"]
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models["denoiser"],
                noise=noise,
                **args,
                steps=50,
                cfg_strength=3.0,
                verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            "sample_gt": {"value": sample_gt, "type": "sample"},
            "sample": {"value": sample, "type": "sample"},
        }
        sample_dict.update(
            dict_reduce(
                cond_vis,
                None,
                {
                    "value": lambda x: torch.cat(x, dim=0),
                    "type": lambda x: x[0],
                },
            )
        )

        return sample_dict

    def _setup_validation(self, **kwargs):
        """Setup flow matching specific validation (latent space MSE) - single GPU only."""
        super()._setup_validation(**kwargs)

        if not self.validation_config.get("enabled", False) or not self.is_master:
            return

        print("Validation config:")
        print(f"  Metric: MSE in latent space")
        print(f"  Data dir: {self.validation_config.get('data_dir', 'N/A')}")

    @torch.no_grad()
    def run_validation(self):
        """Run validation by computing MSE in latent space (master rank only)."""
        if not self.validation_config.get("enabled", False):
            return None

        if not self.is_master:
            return None

        print(f"\nRunning validation on {len(self.val_dataset)} samples...")

        sampler = self.get_sampler()

        total_mse = 0.0
        valid_count = 0

        for batch_idx, batch in enumerate(self.val_dataloader):
            batch_mse = 0.0
            batch_valid_count = 0

            try:
                batch = recursive_to_device(batch, self.device)
                batch_size = batch["x_0"].shape[0]

                noise = torch.randn_like(batch["x_0"])
                exclude_keys = {"x_0", "gt_prob"}
                cond_data = {k: v for k, v in batch.items() if k not in exclude_keys}
                inference_cond = self.get_inference_cond(**cond_data)
                sample_result = sampler.sample(
                    self.models["denoiser"],
                    noise=noise,
                    **inference_cond,
                    steps=50,
                    cfg_strength=3.0,
                    verbose=False,
                )
                z_pred = sample_result.samples
                z_gt = batch["x_0"]

                mse = F.mse_loss(z_pred, z_gt)

                batch_mse = mse.item() * batch_size
                batch_valid_count = batch_size

            except Exception as e:
                print(f"Warning: Error in batch {batch_idx}: {e}")
                import traceback

                traceback.print_exc()
                batch_mse = 0.0
                batch_valid_count = 0

            total_mse += batch_mse
            valid_count += batch_valid_count

        avg_mse = total_mse / valid_count if valid_count > 0 else 0.0

        print(f"  Validation MSE: {avg_mse:.6f} ({valid_count} samples)")

        return {"mse": avg_mse}


class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """

    pass


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        num_workers (int): Number of workers for dataloader (default: 4).
    """

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(self, x_0: sp.SparseTensor, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        mse_per_instance = np.array(
            [
                F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
                for i in range(x_0.shape[0])
            ]
        )
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {
                k: v[:batch].cuda() if not isinstance(v, list) else v[:batch]
                for k, v in data.items()
            }
            noise = data["x_0"].replace(torch.randn_like(data["x_0"].feats))
            sample_gt.append(data["x_0"])
            cond_vis.append(self.vis_cond(**data))
            del data["x_0"]
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models["denoiser"],
                noise=noise,
                **args,
                steps=50,
                cfg_strength=3.0,
                verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)
        sample_dict = {
            "sample_gt": {"value": sample_gt, "type": "sample"},
            "sample": {"value": sample, "type": "sample"},
        }
        sample_dict.update(
            dict_reduce(
                cond_vis,
                None,
                {
                    "value": lambda x: torch.cat(x, dim=0),
                    "type": lambda x: x[0],
                },
            )
        )

        return sample_dict

    def _setup_validation(self, **kwargs):
        """Setup SLAT-specific validation (no decoder needed, latent space metrics)."""
        from .base import Trainer

        Trainer._setup_validation(self, **kwargs)

        if not self.validation_config.get("enabled", False) or not self.is_master:
            return

        print("Validation config:")
        print(f"  Metric: MSE in latent space")
        print(f"  Data dir: {self.validation_config.get('data_dir', 'N/A')}")

    @torch.no_grad()
    def run_validation(self):
        """Run validation by computing MSE in latent space (master rank only)."""
        if not self.validation_config.get("enabled", False):
            return None

        if not self.is_master:
            return None

        print(f"\nRunning validation on {len(self.val_dataset)} samples...")

        sampler = self.get_sampler()

        total_mse = 0.0
        valid_count = 0

        for batch_idx, batch in enumerate(self.val_dataloader):
            batch_mse = 0.0
            batch_valid_count = 0

            try:
                batch = recursive_to_device(batch, self.device)
                batch_size = batch["x_0"].shape[0]

                noise = batch["x_0"].replace(torch.randn_like(batch["x_0"].feats))
                cond_data = {k: v for k, v in batch.items() if k != "x_0"}
                inference_cond = self.get_inference_cond(**cond_data)
                sample_result = sampler.sample(
                    self.models["denoiser"],
                    noise=noise,
                    **inference_cond,
                    steps=50,
                    cfg_strength=3.0,
                    verbose=False,
                )
                z_pred = sample_result.samples
                z_gt = batch["x_0"]

                mse = F.mse_loss(z_pred.feats, z_gt.feats)

                batch_mse = mse.item() * batch_size
                batch_valid_count = batch_size

            except Exception as e:
                print(f"Warning: Error in batch {batch_idx}: {e}")
                import traceback

                traceback.print_exc()
                batch_mse = 0.0
                batch_valid_count = 0

            total_mse += batch_mse
            valid_count += batch_valid_count

        avg_mse = total_mse / valid_count if valid_count > 0 else 0.0

        print(f"  Validation MSE: {avg_mse:.6f} ({valid_count} samples)")

        return {"mse": avg_mse}


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """

    pass
