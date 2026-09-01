import torch
import torch.nn.functional as F


class SparseMaskLoss(torch.nn.Module):
    """
    Sparse Mask Loss for variable-length predictions and targets.

    Combined Dice Loss and Cross Entropy Loss for mask prediction,
    operating on lists of tensors with different lengths per sample.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, pred_list: list[torch.Tensor], target_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            pred_list: List of prediction logits, one tensor per sample
                      Each tensor has shape [N_voxels_i]
            target_list: List of target heatmaps, one tensor per sample
                        Each tensor has shape [N_voxels_i]

        Returns:
            Combined CE + Dice loss
        """
        assert len(pred_list) == len(target_list), "pred and target lists must have same length"

        ce_losses = []
        dice_losses = []

        for pred, target in zip(pred_list, target_list):
            assert (
                pred.shape == target.shape
            ), f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

            pred_prob = torch.sigmoid(pred)

            ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
            ce_losses.append(ce_loss)

            numerator = 2 * (pred_prob * target).sum()
            denominator = pred_prob.sum() + target.sum()
            dice_loss = 1 - (numerator + self.eps) / (denominator + self.eps)
            dice_losses.append(dice_loss)

        ce_loss_avg = torch.stack(ce_losses).mean()
        dice_loss_avg = torch.stack(dice_losses).mean()

        return ce_loss_avg + dice_loss_avg


class SparseFocalMaskLoss(torch.nn.Module):
    """
    Sparse Focal Mask Loss for variable-length predictions and targets.

    Focal Loss combined with balanced Dice Loss for heatmap prediction,
    operating on lists of tensors with different lengths per sample.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(
        self, pred_list: list[torch.Tensor], target_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            pred_list: List of prediction logits, one tensor per sample
                      Each tensor has shape [N_voxels_i]
            target_list: List of target heatmaps, one tensor per sample
                        Each tensor has shape [N_voxels_i]

        Returns:
            Combined Focal + Balanced Dice loss
        """
        assert len(pred_list) == len(target_list), "pred and target lists must have same length"

        focal_losses = []
        dice_losses = []

        for pred, target in zip(pred_list, target_list):
            assert (
                pred.shape == target.shape
            ), f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

            pred_prob = torch.sigmoid(pred)

            neg_class_loss = -(1 - self.alpha) * (
                pred_prob**self.gamma * (1 - target) * torch.log(1 - pred_prob + self.eps)
            )

            pos_class_loss = -self.alpha * (
                (1 - pred_prob) ** self.gamma * target * torch.log(pred_prob + self.eps)
            )

            focal_loss = torch.mean(neg_class_loss + pos_class_loss)
            focal_losses.append(focal_loss)

            pos_intersection = torch.sum(pred_prob * target)
            pos_cardinality = torch.sum(pred_prob + target)
            pos_dice = (pos_intersection + self.eps) / (pos_cardinality + self.eps)

            neg_intersection = torch.sum((1 - pred_prob) * (1 - target))
            neg_cardinality = torch.sum(2 - pred_prob - target)
            neg_dice = (neg_intersection + self.eps) / (neg_cardinality + self.eps)

            dice_loss = 1.5 - pos_dice - neg_dice
            dice_losses.append(dice_loss)

        focal_loss_avg = torch.stack(focal_losses).mean()
        dice_loss_avg = torch.stack(dice_losses).mean()

        return focal_loss_avg + dice_loss_avg


class SparseAffordanceLoss(torch.nn.Module):
    """``SparseMaskLoss`` plus terms aligned with the reported metrics.

    BCE and Dice constrain the heatmap pointwise and in aggregate overlap, but
    neither matches how the affordance metrics score a prediction:

    - ``sim_weight`` penalises ``1 - SIM``. SIM normalises both maps to sum 1,
      so it scores where the mass sits, not how much there is.
    - ``iou_weight`` penalises ``1 - soft IoU`` against the target binarised at
      0.5, which is the form aIoU sweeps thresholds over. Dice uses the raw
      heatmap and the harmonic-mean form instead.
    - ``mae_weight`` adds an L1 term; off by default, since BCE already covers
      pointwise error.

    ``base`` picks the underlying mask loss: ``"mask"`` for BCE + Dice or
    ``"focal"`` for the focal + balanced-Dice variant, which down-weights
    confident voxels and so keeps predictions softer.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        sim_weight: float = 1.0,
        iou_weight: float = 0.0,
        mae_weight: float = 0.0,
        target_threshold: float = 0.5,
        base: str = "mask",
        gamma: float = 2.0,
        alpha: float = 0.25,
    ):
        super().__init__()
        self.eps = eps
        self.sim_weight = sim_weight
        self.iou_weight = iou_weight
        self.mae_weight = mae_weight
        self.target_threshold = target_threshold
        if base == "mask":
            self.mask_loss = SparseMaskLoss(eps=eps)
        elif base == "focal":
            self.mask_loss = SparseFocalMaskLoss(gamma=gamma, alpha=alpha, eps=eps)
        else:
            raise ValueError(f"Unknown base loss: {base}")

    def forward(
        self, pred_list: list[torch.Tensor], target_list: list[torch.Tensor]
    ) -> torch.Tensor:
        sim_losses, iou_losses, mae_losses = [], [], []

        for pred, target in zip(pred_list, target_list):
            pred_prob = torch.sigmoid(pred)

            if self.sim_weight:
                pred_normalized = pred_prob / (pred_prob.sum() + self.eps)
                target_normalized = target / (target.sum() + self.eps)
                sim = torch.minimum(pred_normalized, target_normalized).sum()
                sim_losses.append(1 - sim)

            if self.iou_weight:
                target_binary = (target > self.target_threshold).to(pred_prob.dtype)
                intersection = (pred_prob * target_binary).sum()
                union = pred_prob.sum() + target_binary.sum() - intersection
                iou_losses.append(1 - (intersection + self.eps) / (union + self.eps))

            if self.mae_weight:
                mae_losses.append((pred_prob - target).abs().mean())

        loss = self.mask_loss(pred_list, target_list)
        if sim_losses:
            loss = loss + self.sim_weight * torch.stack(sim_losses).mean()
        if iou_losses:
            loss = loss + self.iou_weight * torch.stack(iou_losses).mean()
        if mae_losses:
            loss = loss + self.mae_weight * torch.stack(mae_losses).mean()
        return loss
