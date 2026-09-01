"""
Affordance grounding metrics.

Two evaluation regimes:

* **Point level** (complete-input protocol) — predictions are transferred to the
  raw annotation point cloud and compared against the continuous ground-truth
  heatmap: ``AverageIoU`` (mean IoU over 20 thresholds, GT binarised at 0.5),
  ``AUC``, ``Similarity`` (SIM) and ``TotalMAE``. ``AffordanceMetrics`` computes
  all four in a single pass and is what the trainer uses for validation.
* **Voxel level** (partial-input protocol) — predicted and ground-truth voxels
  live on the same canonical 64^3 grid but cover different sets, so both sides
  are thresholded and compared as coordinate sets: ``VoxelAffordanceMetrics``
  returns ``aiou`` (set IoU) and ``acd`` (bidirectional Chamfer distance),
  each averaged over thresholds 0.1-0.5.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAUROC

__all__ = [
    "AverageIoU",
    "AUC",
    "Similarity",
    "TotalMAE",
    "AffordanceMetrics",
    "VoxelAffordanceMetrics",
]


class AverageIoU(nn.Module):
    """
    Computes Average Intersection over Union (IoU) across multiple thresholds.

    This metric calculates IoU at multiple thresholds (0 to 1) and averages them,
    similar to how Average Precision works. This provides a more robust evaluation
    than using a single threshold.

    Supports variable-length predictions and targets as List[torch.Tensor].
    """

    def __init__(self, num_thresholds: int = 20, target_threshold: float = 0.5):
        """
        Initialize AverageIoU metric.

        Args:
            num_thresholds: Number of thresholds to use between 0 and 1 (default: 20)
            target_threshold: Threshold to binarize target values (default: 0.5)
        """
        super().__init__()
        self.num_thresholds = num_thresholds
        self.target_threshold = target_threshold

    def forward(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Compute Average IoU for variable-length predictions and targets.

        Args:
            preds: List of prediction logits, one tensor per sample
                   Each tensor has shape [N_voxels_i]
            targets: List of target heatmaps, one tensor per sample
                    Each tensor has shape [N_voxels_i]
            return_per_sample: If True, return per-sample IoUs as well

        Returns:
            Average IoU across all samples and thresholds
            If return_per_sample=True, returns (avg_iou, per_sample_ious)
        """
        assert len(preds) == len(targets), "preds and targets must have same length"

        device = preds[0].device
        thresholds = torch.linspace(0, 1, self.num_thresholds, device=device)

        sample_ious = []

        for pred_logits, target in zip(preds, targets):
            pred_probs = torch.sigmoid(pred_logits)

            target_binary = (target >= self.target_threshold).float()

            if target_binary.sum() == 0:
                continue

            iou_values = []
            for threshold in thresholds:
                pred_mask = (pred_probs >= threshold).float()

                intersection = torch.logical_and(pred_mask.bool(), target_binary.bool()).sum()
                union = torch.logical_or(pred_mask.bool(), target_binary.bool()).sum()

                if union > 0:
                    iou = intersection.float() / union.float()
                else:
                    iou = torch.tensor(0.0, device=device)

                iou_values.append(iou)

            sample_avg_iou = torch.stack(iou_values).mean()
            sample_ious.append(sample_avg_iou)

        if len(sample_ious) == 0:
            avg_iou = torch.tensor(0.0, device=device)
        else:
            avg_iou = torch.stack(sample_ious).mean()

        if return_per_sample:
            return avg_iou, sample_ious
        return avg_iou


class AUC(nn.Module):
    """
    Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) for affordance prediction.

    Implements a metric that computes the AUC score between predicted heatmaps and ground truth heatmaps.
    Uses torchmetrics.BinaryAUROC internally and averages across samples.

    Supports variable-length predictions and targets as List[torch.Tensor].
    """

    def __init__(self, target_threshold: float = 0.5):
        """
        Initialize AUC metric.

        Args:
            target_threshold: Threshold to binarize target values (default: 0.5)
        """
        super().__init__()
        self.target_threshold = target_threshold
        self.auroc = BinaryAUROC(thresholds=None)

    def forward(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Compute AUC for variable-length predictions and targets.

        Args:
            preds: List of prediction logits, one tensor per sample
                   Each tensor has shape [N_voxels_i]
            targets: List of target heatmaps, one tensor per sample
                    Each tensor has shape [N_voxels_i]
            return_per_sample: If True, return per-sample AUCs as well

        Returns:
            Average AUC across all valid samples
            If return_per_sample=True, returns (avg_auc, per_sample_aucs)
        """
        assert len(preds) == len(targets), "preds and targets must have same length"

        device = preds[0].device
        sample_aucs = []

        for pred_logits, target in zip(preds, targets):
            pred_probs = torch.sigmoid(pred_logits)

            target_binary = (target >= self.target_threshold).float()

            if target_binary.sum() == 0:
                continue

            if target_binary.sum() == target_binary.numel():
                continue

            try:
                self.auroc.reset()
                sample_auc = self.auroc(pred_probs, target_binary.long())
                sample_aucs.append(sample_auc)
            except Exception:
                continue

        if len(sample_aucs) == 0:
            avg_auc = torch.tensor(0.0, device=device)
        else:
            avg_auc = torch.stack(sample_aucs).mean()

        if return_per_sample:
            return avg_auc, sample_aucs
        return avg_auc


class Similarity(nn.Module):
    """
    Custom metric for calculating similarity between predicted and target heatmaps.

    SIM normalizes the maps and computes the sum of the element-wise minimums.

    Supports variable-length predictions and targets as List[torch.Tensor].
    """

    def __init__(self, eps: float = 1e-12):
        """
        Initialize Similarity metric.

        Args:
            eps: Small epsilon value to avoid division by zero (default: 1e-12)
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Compute Similarity for variable-length predictions and targets.

        Args:
            preds: List of prediction logits, one tensor per sample
                   Each tensor has shape [N_voxels_i]
            targets: List of target heatmaps, one tensor per sample
                    Each tensor has shape [N_voxels_i]
            return_per_sample: If True, return per-sample similarities as well

        Returns:
            Average similarity across all samples
            If return_per_sample=True, returns (avg_sim, per_sample_sims)
        """
        assert len(preds) == len(targets), "preds and targets must have same length"

        device = preds[0].device
        sample_sims = []

        for pred_logits, target in zip(preds, targets):
            pred_probs = torch.sigmoid(pred_logits)

            pred_sum = pred_probs.sum() + self.eps
            target_sum = target.sum() + self.eps

            pred_normalized = pred_probs / pred_sum
            target_normalized = target / target_sum

            intersection = torch.minimum(pred_normalized, target_normalized)

            similarity = intersection.sum()
            sample_sims.append(similarity)

        if len(sample_sims) == 0:
            avg_sim = torch.tensor(0.0, device=device)
        else:
            avg_sim = torch.stack(sample_sims).mean()

        if return_per_sample:
            return avg_sim, sample_sims
        return avg_sim


class TotalMAE(nn.Module):
    """
    Custom metric for calculating Mean Absolute Error across all voxels in point clouds.

    This metric accumulates absolute error sum and total number of voxels
    to properly calculate MAE for 3D affordance heatmap data.

    Supports variable-length predictions and targets as List[torch.Tensor].
    """

    def __init__(self):
        """Initialize TotalMAE metric."""
        super().__init__()

    def forward(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Compute Mean Absolute Error for variable-length predictions and targets.

        Args:
            preds: List of prediction logits, one tensor per sample
                   Each tensor has shape [N_voxels_i]
            targets: List of target heatmaps, one tensor per sample
                    Each tensor has shape [N_voxels_i]
            return_per_sample: If True, return per-sample MAEs as well

        Returns:
            Mean Absolute Error across all voxels
            If return_per_sample=True, returns (total_mae, per_sample_maes)
        """
        assert len(preds) == len(targets), "preds and targets must have same length"

        device = preds[0].device
        total_error = torch.tensor(0.0, device=device)
        total_voxels = 0
        sample_maes = []

        for pred_logits, target in zip(preds, targets):
            pred_probs = torch.sigmoid(pred_logits)

            abs_error = torch.abs(pred_probs - target)

            sample_mae = abs_error.mean()
            sample_maes.append(sample_mae)
            total_error += abs_error.sum()
            total_voxels += target.numel()

        if total_voxels == 0:
            total_mae = torch.tensor(0.0, device=device)
        else:
            total_mae = total_error / total_voxels

        if return_per_sample:
            return total_mae, sample_maes
        return total_mae


def _roc_auc(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute AUC (Area Under ROC Curve) efficiently in PyTorch without state accumulation.

    Uses TPR (True Positive Rate) and false positive rate to compute ROC-AUC.

    Args:
        preds: Predicted probabilities [N]
        targets: Binary targets [N]

    Returns:
        AUC score as scalar tensor
    """
    sorted_indices = torch.argsort(preds, descending=True)
    sorted_targets = targets[sorted_indices]

    n_pos = sorted_targets.sum()
    n_neg = sorted_targets.numel() - n_pos

    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.5, device=preds.device)

    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1 - sorted_targets, dim=0)

    tp = torch.cat([torch.zeros(1, device=tp.device), tp])
    fp = torch.cat([torch.zeros(1, device=fp.device), fp])

    tpr = tp / n_pos
    false_pos_rate = fp / n_neg

    auc = torch.trapezoid(tpr, false_pos_rate)

    return auc


class AffordanceMetrics(nn.Module):
    """
    Efficiently computes multiple affordance metrics in a single pass.

    Metrics computed:
    - avg_iou: Average IoU across multiple thresholds
    - auc: Area Under ROC Curve
    - sim: Normalized similarity
    - mae: Mean Absolute Error

    Key efficiency: Sigmoid transformation and sample iteration performed only once.
    All metrics computed from the same probability predictions.

    Args:
        num_thresholds: Number of thresholds for Average IoU (default: 20)
        target_threshold: Threshold to binarize target values (default: 0.5)
        eps: Small epsilon value to avoid division by zero (default: 1e-12)
    """

    def __init__(
        self,
        num_thresholds: int = 20,
        target_threshold: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.target_threshold = target_threshold
        self.eps = eps

    def forward(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute all affordance metrics in a single pass (vectorized).

        Args:
            preds: List of prediction logits, one tensor per sample
                   Each tensor has shape [N_voxels_i]
            targets: List of target heatmaps, one tensor per sample
                    Each tensor has shape [N_voxels_i]

        Returns:
            Dictionary with keys: 'avg_iou', 'auc', 'sim', 'mae'
            Each value is a scalar tensor
        """
        assert len(preds) == len(targets), "preds and targets must have same length"

        if len(preds) == 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return {
                "avg_iou": torch.tensor(0.0, device=device),
                "auc": torch.tensor(0.0, device=device),
                "sim": torch.tensor(0.0, device=device),
                "mae": torch.tensor(0.0, device=device),
            }

        device = preds[0].device
        batch_size = len(preds)

        lengths = torch.tensor([p.numel() for p in preds], device=device)
        max_len = lengths.max().item()

        padded_preds = torch.zeros(batch_size, max_len, device=device)
        padded_targets = torch.zeros(batch_size, max_len, device=device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        for i, (p, t) in enumerate(zip(preds, targets)):
            length = p.numel()
            padded_preds[i, :length] = p
            padded_targets[i, :length] = t
            mask[i, :length] = True


        pred_probs = torch.sigmoid(padded_preds)

        target_binary = (padded_targets >= self.target_threshold).float()

        pred_probs_masked = pred_probs * mask
        target_binary_masked = target_binary * mask
        padded_targets_masked = padded_targets * mask

        thresholds = torch.linspace(0, 1, self.num_thresholds, device=device)

        pred_expanded = pred_probs_masked.unsqueeze(-1)
        thresh_expanded = thresholds.view(1, 1, -1)

        pred_masks = (pred_expanded >= thresh_expanded).float()

        target_expanded = target_binary_masked.unsqueeze(-1)
        mask_expanded = mask.unsqueeze(-1)

        pred_masks = pred_masks * mask_expanded

        intersection = (pred_masks * target_expanded).sum(dim=1)
        union = ((pred_masks + target_expanded) > 0).float().sum(dim=1)

        iou_per_thresh = torch.where(union > 0, intersection / union, torch.zeros_like(union))

        target_sums = target_binary_masked.sum(dim=1)
        valid_samples = target_sums > 0

        sample_avg_iou = iou_per_thresh.mean(dim=1)

        if valid_samples.any():
            avg_iou = sample_avg_iou[valid_samples].mean()
        else:
            avg_iou = torch.tensor(0.0, device=device)

        sample_aucs = []
        for i in range(batch_size):
            length = lengths[i].item()
            if length == 0:
                continue
            pred_i = pred_probs[i, :length]
            target_i = target_binary[i, :length]

            target_sum = target_i.sum()
            if target_sum == 0 or target_sum == length:
                continue

            try:
                sample_auc = _roc_auc(pred_i, target_i)
                sample_aucs.append(sample_auc)
            except Exception:
                pass

        auc = (
            torch.stack(sample_aucs).mean()
            if len(sample_aucs) > 0
            else torch.tensor(0.0, device=device)
        )

        pred_sums = pred_probs_masked.sum(dim=1) + self.eps
        target_sums_sim = padded_targets_masked.sum(dim=1) + self.eps

        pred_normalized = pred_probs_masked / pred_sums.unsqueeze(1)
        target_normalized = padded_targets_masked / target_sums_sim.unsqueeze(1)

        intersection_sim = torch.minimum(pred_normalized, target_normalized).sum(dim=1)

        sim = intersection_sim.mean()

        abs_error = torch.abs(pred_probs_masked - padded_targets_masked)
        total_abs_error = abs_error.sum()
        total_voxels = lengths.sum()

        mae = (
            total_abs_error / total_voxels
            if total_voxels > 0
            else torch.tensor(0.0, device=device)
        )

        return {
            "avg_iou": avg_iou,
            "auc": auc,
            "sim": sim,
            "mae": mae,
        }


class VoxelAffordanceMetrics(nn.Module):
    """
    Threshold-based affordance metrics using voxel/point cloud comparison.

    This metric thresholds both GT and predicted affordance heatmaps at multiple
    thresholds (0.1 to 0.5), then computes:
    - Affo-aIoU: Average IoU across valid thresholds (set-based on voxel coordinates)
    - Affo-aCD: Average Chamfer Distance across valid thresholds (world coordinates)

    Valid thresholds are those where GT has at least one positive voxel after thresholding.
    This handles datasets with continuous affordance scores below 0.5.
    """

    def __init__(
        self,
        thresholds: list[float] = None,
        voxel_resolution: int = 64,
    ):
        """
        Initialize VoxelAffordanceMetrics.

        Args:
            thresholds: List of thresholds for binarizing affordances
                       (default: [0.1, 0.2, 0.3, 0.4, 0.5])
            voxel_resolution: Voxel grid resolution (default: 64)
        """
        super().__init__()
        self.thresholds = thresholds if thresholds is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
        self.voxel_resolution = voxel_resolution

    def voxel_to_3d_coords(self, voxel_coords: np.ndarray) -> np.ndarray:
        """
        Convert voxel indices to 3D coordinates in [-0.5, 0.5] range.

        Args:
            voxel_coords: (N, 3) array of voxel indices [0, resolution)

        Returns:
            (N, 3) array of 3D coordinates in [-0.5, 0.5]
        """
        return ((voxel_coords + 0.5) / self.voxel_resolution - 0.5).astype(np.float32)

    def _volumetric_iou(
        self,
        pred_sparse_structure: np.ndarray,
        pred_affordance_logits: np.ndarray,
        gt_affordance_path: Path,
        query_idx: int,
    ) -> float:
        """
        Compute affordance volumetric Average IoU across multiple thresholds.

        Thresholds both GT and predicted affordance heatmaps at multiple thresholds
        (self.thresholds), then computes average IoU using voxel-level set operations.
        Only includes thresholds where GT has at least one positive voxel.

        Args:
            pred_sparse_structure: (N_pred, 3) predicted voxel coordinates [x, y, z]
            pred_affordance_logits: (N_pred,) predicted affordance logits
            gt_affordance_path: Path to GT affordance NPZ file (affordances/{sha256}/affordance.npz)
            query_idx: Query index (0-4)

        Returns:
            Average IoU value in range [0, 1], or None if computation fails
        """
        try:
            affo_data = np.load(str(gt_affordance_path), allow_pickle=True)
            gt_voxel_coords = affo_data["coords"].astype(np.int32)
            gt_heatmap = affo_data["heatmap"][:, query_idx]

            pred_probs = 1.0 / (1.0 + np.exp(-pred_affordance_logits))

            ious = []
            for threshold in self.thresholds:
                gt_affordance_mask = gt_heatmap >= threshold

                if gt_affordance_mask.sum() == 0:
                    continue

                gt_affordance_voxels = gt_voxel_coords[gt_affordance_mask]

                pred_affordance_mask = pred_probs >= threshold
                pred_affordance_voxels = pred_sparse_structure[pred_affordance_mask]

                pred_set = set(map(tuple, pred_affordance_voxels))
                gt_set = set(map(tuple, gt_affordance_voxels))

                intersection = len(pred_set & gt_set)
                union = len(pred_set | gt_set)

                if union > 0:
                    ious.append(intersection / union)

            return np.mean(ious) if len(ious) > 0 else 0.0

        except (KeyError, ValueError, OSError) as e:
            print(f"Skipping IoU for {gt_affordance_path}: {e}")
            return None

    def _chamfer_distance(
        self,
        pred_sparse_structure: np.ndarray,
        pred_affordance_logits: np.ndarray,
        gt_affordance_path: Path,
        query_idx: int,
    ) -> float:
        """
        Compute affordance Average Chamfer Distance across multiple thresholds.

        Thresholds both GT and predicted affordance heatmaps at multiple thresholds
        (self.thresholds), converts to 3D world coordinates, then computes average
        bidirectional Chamfer Distance. Only includes thresholds where GT has at least
        one positive voxel.

        Args:
            pred_sparse_structure: (N_pred, 3) predicted voxel coordinates [x, y, z]
            pred_affordance_logits: (N_pred,) predicted affordance logits
            gt_affordance_path: Path to GT affordance NPZ file
            query_idx: Query index (0-4)

        Returns:
            Average Chamfer Distance value, or None if computation fails
        """
        try:
            affo_data = np.load(str(gt_affordance_path), allow_pickle=True)
            gt_coords = affo_data["coords"].astype(np.float32)
            gt_world = (gt_coords + 0.5) / self.voxel_resolution - 0.5
            gt_heatmap = affo_data["heatmap"][:, query_idx]

            pred_probs = 1.0 / (1.0 + np.exp(-pred_affordance_logits))

            chamfer_distances = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for threshold in self.thresholds:
                gt_affordance_mask = gt_heatmap >= threshold

                if gt_affordance_mask.sum() == 0:
                    continue

                gt_affordance_world = gt_world[gt_affordance_mask]

                pred_affordance_mask = pred_probs >= threshold
                pred_affordance_voxels = pred_sparse_structure[pred_affordance_mask]

                if len(pred_affordance_voxels) == 0:
                    continue

                pred_affordance_world = self.voxel_to_3d_coords(pred_affordance_voxels)

                gt_pts = torch.from_numpy(gt_affordance_world).float().to(device)
                pred_pts = torch.from_numpy(pred_affordance_world).float().to(device)

                distances = torch.cdist(pred_pts, gt_pts)

                pred_to_gt = distances.min(dim=1)[0]
                pred_to_gt_mean = pred_to_gt.mean().item()

                gt_to_pred = distances.min(dim=0)[0]
                gt_to_pred_mean = gt_to_pred.mean().item()

                cd = (pred_to_gt_mean + gt_to_pred_mean) / 2.0
                chamfer_distances.append(cd)

            return np.mean(chamfer_distances) if len(chamfer_distances) > 0 else None

        except (KeyError, ValueError, OSError) as e:
            print(f"Skipping Chamfer distance for {gt_affordance_path}: {e}")
            return None

    def forward(
        self,
        pred_sparse_structure_list: list[np.ndarray],
        pred_logits_list: list[np.ndarray],
        gt_affordance_paths: list[Path],
        query_indices: list[int],
        return_per_sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute threshold-based affordance metrics.

        Args:
            pred_sparse_structure_list: List of (N_pred_i, 3) voxel coordinate arrays
            pred_logits_list: List of (N_pred_i,) affordance logit arrays
            gt_affordance_paths: List of paths to GT affordance NPZ files
            query_indices: List of query indices (0-4)
            return_per_sample: If True, return per-sample metrics as well

        Returns:
            Dictionary containing:
            - 'aiou': Affordance volumetric IoU
            - 'acd': Affordance Chamfer Distance
            If return_per_sample=True, also includes:
            - 'aiou_per_sample', 'acd_per_sample'
        """
        assert len(pred_sparse_structure_list) == len(pred_logits_list)
        assert len(pred_sparse_structure_list) == len(gt_affordance_paths)
        assert len(pred_sparse_structure_list) == len(query_indices)

        sample_ious = []
        sample_cds = []

        for pred_coords, pred_logits, gt_affo_path, query_idx in zip(
            pred_sparse_structure_list,
            pred_logits_list,
            gt_affordance_paths,
            query_indices,
        ):
            sample_ious.append(
                self._volumetric_iou(
                    pred_coords, pred_logits, gt_affo_path, query_idx
                )
            )
            sample_cds.append(
                self._chamfer_distance(
                    pred_coords, pred_logits, gt_affo_path, query_idx
                )
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ious = np.array([np.nan if v is None else v for v in sample_ious], dtype=np.float64)
        cds = np.array([np.nan if v is None else v for v in sample_cds], dtype=np.float64)
        results = {
            "aiou": torch.tensor(float(np.nanmean(ious)) if np.isfinite(ious).any() else 0.0, device=device),
            "acd": torch.tensor(float(np.nanmean(cds)) if np.isfinite(cds).any() else float("inf"), device=device),
        }

        if return_per_sample:
            results["aiou_per_sample"] = [torch.tensor(v, device=device) for v in ious]
            results["acd_per_sample"] = [torch.tensor(v, device=device) for v in cds]

        return results
