"""
Stage-2 trainer: text-conditioned affordance heatmap flow.

Flow matching runs in logit space over the ground-truth voxels of an object;
the loss is applied to the x0 prediction (BCE + Dice by default, focal
optionally) and validation reports mean-threshold IoU on the test split.
"""
from typing import *
import copy
import traceback

import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModel

from .flow_matching import SparseFlowMatchingCFGTrainer
from ..metrics import AffordanceMetrics
from ..models.losses import SparseAffordanceLoss, SparseFocalMaskLoss, SparseMaskLoss
from ..modules import sparse as sp
from ..utils import distributed
from ..utils.data import recursive_to_device


class TextConditionedMixin:
    """
    Mixin for text-conditioned models.

    Args:
        text_cond_model: The text conditioning model.
    """

    def __init__(self, *args, text_cond_model: str = "openai/clip-vit-large-patch14", **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cond_model_name = text_cond_model
        self.text_cond_model = None
        self._use_open_clip = False

    def _init_text_cond_model(self):
        """
        Initialize the text conditioning model.
        """
        with distributed.local_master_first():
            if self.text_cond_model_name.startswith("hf-hub:"):
                import open_clip

                model, _ = open_clip.create_model_from_pretrained(self.text_cond_model_name)
                tokenizer = open_clip.get_tokenizer(self.text_cond_model_name)
                self._use_open_clip = True
            else:
                model = CLIPTextModel.from_pretrained(self.text_cond_model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.text_cond_model_name)
                self._use_open_clip = False
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            "model": model,
            "tokenizer": tokenizer,
        }
        self.text_cond_model["null_cond"] = self.encode_text([""])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and isinstance(
            text[0], str
        ), "TextConditionedMixin only supports list of strings as cond"
        if self.text_cond_model is None:
            self._init_text_cond_model()

        if self._use_open_clip:
            tokens = self.text_cond_model["tokenizer"](text).cuda()
            embeddings = self.text_cond_model["model"].encode_text(tokens, normalize=False)
            embeddings = embeddings.unsqueeze(1)
        else:
            encoding = self.text_cond_model["tokenizer"](
                text, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokens = encoding["input_ids"].cuda()
            embeddings = self.text_cond_model["model"](input_ids=tokens).last_hidden_state

        return embeddings

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_text(cond)
        kwargs["neg_cond"] = self.text_cond_model["null_cond"].repeat(cond.shape[0], 1, 1)
        cond = super().get_cond(cond, **kwargs)
        return cond

    def get_inference_cond(self, cond, neg_cond_mode="zeros", **kwargs):
        """
        Get the conditioning data for inference.

        ``neg_cond_mode`` mirrors the inference pipelines: ``"zeros"`` for an
        all-zero negative condition, ``"empty_text"`` for the CLIP embedding of
        the empty string.
        """
        cond = self.encode_text(cond)
        if neg_cond_mode == "zeros":
            kwargs["neg_cond"] = torch.zeros_like(cond)
        elif neg_cond_mode == "empty_text":
            kwargs["neg_cond"] = self.text_cond_model["null_cond"].repeat(cond.shape[0], 1, 1)
        else:
            raise ValueError(f"Unknown neg_cond_mode: {neg_cond_mode}")
        cond = super().get_inference_cond(cond, **kwargs)
        return cond


class AffordanceFlowTrainer(
    TextConditionedMixin, SparseFlowMatchingCFGTrainer
):
    """
    Text-conditioned Affordance Heatmap Flow Matching Trainer with Classifier-Free Guidance.

    This trainer performs flow matching in logit space for affordance heatmap generation.
    Key differences from standard SLAT flow matching:
    - Uses SparseMaskLoss (BCE + Dice) or SparseFocalMaskLoss (Focal + Balanced Dice) instead of MSE
    - Operates on 1-channel affordance logits instead of 8-channel SLAT
    - Validates with Average IoU instead of MSE

    Args:
        models (dict[str, nn.Module]): Models to train (contains 'denoiser')
        dataset (torch.utils.data.Dataset): Affordance flow dataset
        output_dir (str): Output directory
        load_dir (str): Load directory
        step (int): Step to load
        batch_size (int): Batch size
        batch_size_per_gpu (int): Batch size per GPU
        batch_split (int): Split batch with gradient accumulation
        max_steps (int): Max steps
        optimizer (dict): Optimizer config
        lr_scheduler (dict): Learning rate scheduler config
        elastic (dict): Elastic memory management config
        grad_clip (float or dict): Gradient clip config
        ema_rate (float or list): Exponential moving average rates
        fp16_mode (str): FP16 mode
        fp16_scale_growth (float): Scale growth for FP16
        i_print (int): Print interval
        i_log (int): Log interval
        i_sample (int): Sample interval
        i_save (int): Save interval
        i_ddpcheck (int): DDP check interval
        t_schedule (dict): Time schedule for flow matching
        sigma_min (float): Minimum noise level
        p_uncond (float): Probability of dropping conditions (for CFG)
        text_cond_model (str): Text conditioning model (CLIP)
        noise_scale (float): Noise scaling for logit space (default: 5.0)
        loss_config (dict): Loss configuration. If not specified, defaults to SparseMaskLoss.
            Example: {"name": "SparseFocalMaskLoss", "gamma": 2.0, "alpha": 0.25}
    """

    def __init__(
        self,
        *args,
        noise_scale: float = 5.0,
        loss_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_scale = noise_scale

        if loss_config is None:
            self.mask_loss = SparseMaskLoss()
            self.loss_name = "SparseMaskLoss()"
        else:
            loss_name = loss_config.get("name", "SparseMaskLoss")
            if loss_name == "SparseFocalMaskLoss":
                gamma = loss_config.get("gamma", 2.0)
                alpha = loss_config.get("alpha", 0.25)
                eps = loss_config.get("eps", 1e-6)
                self.mask_loss = SparseFocalMaskLoss(gamma=gamma, alpha=alpha, eps=eps)
                self.loss_name = f"SparseFocalMaskLoss(gamma={gamma}, alpha={alpha}, eps={eps})"
            elif loss_name == "SparseMaskLoss":
                eps = loss_config.get("eps", 1e-6)
                self.mask_loss = SparseMaskLoss(eps=eps)
                self.loss_name = f"SparseMaskLoss(eps={eps})"
            elif loss_name == "SparseAffordanceLoss":
                kwargs = {k: loss_config[k] for k in
                          ("eps", "sim_weight", "iou_weight", "mae_weight",
                           "base", "gamma", "alpha") if k in loss_config}
                self.mask_loss = SparseAffordanceLoss(**kwargs)
                self.loss_name = f"SparseAffordanceLoss({kwargs})"
            else:
                raise ValueError(f"Unknown loss name: {loss_name}")

        self.eval_metric = AffordanceMetrics(
            num_thresholds=20,
            target_threshold=0.5,
            eps=1e-12,
        )

        self.primary_metric = self.validation_config.get("primary_metric", "avg_iou")

        if self.is_master:
            print(f"Affordance Flow Matching Config:")
            print(f"  Noise scale: {self.noise_scale}")
            print(f"  Loss: {self.loss_name}")
            print(f"  Text model: {self.text_cond_model_name}")
            print(f"  Validation metrics: avg_iou, auc, sim, mae")

        self._print_validation_config()

    def _print_validation_config(self):
        """Print validation configuration."""
        if not self.validation_config.get("enabled", False) or not self.is_master:
            return

        print("Validation config:")
        print(f"  Metrics: avg_iou, auc, sim, mae")
        print(f"  Data dir: {self.validation_config.get('data_dir', 'N/A')}")
        print(f"  Primary metric (for checkpointing): {self.primary_metric}")
        print(f"  Metric mode: {self.validation_config.get('metric_mode', 'max')}")
        print(
            f"  Sampler: steps={self.validation_config.get('steps', 50)}, "
            f"cfg={self.validation_config.get('cfg_strength', 1.0)}, "
            f"noise={self.validation_config.get('noise_scale', 0.1)}, "
            f"neg_cond={self.validation_config.get('neg_cond_mode', 'zeros')}"
        )

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond: List[str],
        gt_prob: List[torch.Tensor],
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for affordance flow matching.

        Args:
            x_0: Affordance logits as sparse tensor [B, N_voxels_i, 1]
            cond: List of text queries [B]
            gt_prob: List of ground truth affordance probabilities [B], each [N_voxels_i]
            **kwargs: Additional arguments

        Returns:
            terms: Dict with loss values
            stats: Dict with statistics
        """
        try:
            batch_size = x_0.shape[0]

            noise = x_0.replace(torch.randn_like(x_0.feats) * self.noise_scale)
            t = self.sample_t(batch_size).to(x_0.device).float()
            x_t = self.diffuse(x_0, t, noise=noise)
            cond = self.get_cond(cond, **kwargs)

            pred_v = self.training_models["denoiser"](x_t, t * 1000, cond, **kwargs)
            assert pred_v.shape == noise.shape == x_0.shape

            x_0_pred = noise.replace((1 - self.sigma_min) * noise.feats - pred_v.feats)

            x_0_pred_list = [
                x_0_pred.feats[x_0_pred.layout[i]].squeeze(-1) for i in range(batch_size)
            ]

            terms = edict()
            terms["mask_loss"] = self.mask_loss(x_0_pred_list, gt_prob)
            terms["loss"] = terms["mask_loss"]

            with torch.no_grad():
                try:
                    metrics_dict = self.eval_metric(x_0_pred_list, gt_prob)
                    terms[self.primary_metric] = metrics_dict[self.primary_metric]
                except Exception:
                    pass

            stats = {}

            return terms, stats

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR in training_losses: {e}")
            print(f"{'='*80}")
            traceback.print_exc()
            print(f"{'='*80}\n")
            raise

    @torch.no_grad()
    def run_validation(self):
        """
        Run validation by computing all metrics on validation dataset (master rank only).

        Returns:
            Dictionary with validation metrics: {avg_iou, auc, sim, mae}
        """
        if not self.validation_config.get("enabled", False):
            return None

        if not self.is_master:
            return None

        steps = self.validation_config.get("steps", 50)
        cfg_strength = self.validation_config.get("cfg_strength", 1.0)
        noise_scale = self.validation_config.get("noise_scale", 0.1)
        neg_cond_mode = self.validation_config.get("neg_cond_mode", "zeros")

        print(
            f"\nRunning validation on {len(self.val_dataset)} samples "
            f"(steps={steps}, cfg={cfg_strength}, noise={noise_scale}, neg_cond={neg_cond_mode})..."
        )

        sampler = self.get_sampler()

        device = self.device
        total_avg_iou = torch.tensor(0.0, device=device)
        total_auc = torch.tensor(0.0, device=device)
        total_sim = torch.tensor(0.0, device=device)
        total_mae = torch.tensor(0.0, device=device)
        valid_count = 0

        model = self.models["denoiser"]
        model.eval()

        for batch_idx, batch in enumerate(self.val_dataloader):
            try:
                batch = recursive_to_device(batch, self.device)
                batch_size = batch["x_0"].shape[0]

                noise = batch["x_0"].replace(torch.randn_like(batch["x_0"].feats) * noise_scale)
                cond_data = {k: v for k, v in batch.items() if k not in ["x_0", "gt_prob"]}
                inference_cond = self.get_inference_cond(**cond_data, neg_cond_mode=neg_cond_mode)

                sample_result = sampler.sample(
                    model,
                    noise=noise,
                    **inference_cond,
                    steps=steps,
                    cfg_strength=cfg_strength,
                    verbose=False,
                )
                x_0_pred = sample_result.samples

                x_0_pred_list = [
                    x_0_pred.feats[x_0_pred.layout[i]].squeeze(-1) for i in range(batch_size)
                ]

                metrics_dict = self.eval_metric(x_0_pred_list, batch["gt_prob"])

                total_avg_iou += metrics_dict["avg_iou"] * batch_size
                total_auc += metrics_dict["auc"] * batch_size
                total_sim += metrics_dict["sim"] * batch_size
                total_mae += metrics_dict["mae"] * batch_size
                valid_count += batch_size

            except Exception as e:
                print(f"Warning: Error in validation batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

        if valid_count > 0:
            avg_iou = (total_avg_iou / valid_count).item()
            auc = (total_auc / valid_count).item()
            sim = (total_sim / valid_count).item()
            mae = (total_mae / valid_count).item()
        else:
            avg_iou = auc = sim = mae = 0.0

        print(f"  Validation Results:")
        print(f"    AVG_IOU: {avg_iou:.4f}")
        print(f"    AUC: {auc:.4f}")
        print(f"    SIM: {sim:.4f}")
        print(f"    MAE: {mae:.4f}")
        print(f"  Valid samples: {valid_count}")

        model.train()

        return {
            "avg_iou": avg_iou,
            "auc": auc,
            "sim": sim,
            "mae": mae,
        }

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Run inference snapshot for visualization.

        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for inference
            verbose: If True, show progress

        Returns:
            Dictionary with predicted and ground truth heatmaps
        """
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        sampler = self.get_sampler()

        all_heatmaps_pred = []
        all_heatmaps_gt = []
        all_queries = []

        model = self.models["denoiser"]
        model.eval()

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = recursive_to_device(data, self.device)

            noise = data["x_0"].replace(torch.randn_like(data["x_0"].feats) * self.noise_scale)
            cond_data = {k: v for k, v in data.items() if k not in ["x_0", "gt_prob"]}
            inference_cond = self.get_inference_cond(**cond_data)

            sample_result = sampler.sample(
                model,
                noise=noise,
                **inference_cond,
                steps=50,
                cfg_strength=3.0,
                verbose=verbose,
            )
            x_0_pred = sample_result.samples

            heatmaps_pred = [
                torch.sigmoid(x_0_pred.feats[x_0_pred.layout[i]].squeeze(-1))
                for i in range(data["x_0"].shape[0])
            ]

            all_heatmaps_pred.extend(heatmaps_pred)
            all_heatmaps_gt.extend(data["gt_prob"])
            all_queries.extend(data["cond"])

        model.train()

        return {
            "heatmaps_pred": all_heatmaps_pred,
            "heatmaps_gt": all_heatmaps_gt,
            "queries": all_queries,
        }
