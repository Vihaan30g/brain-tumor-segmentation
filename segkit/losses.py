import torch
import torch.nn as nn


class BCEWithDiceLoss(nn.Module):
    """BCE-with-logits + soft Dice (per-sample Dice, averaged over the batch).

    The loss is always computed in float32, even when the model runs under AMP.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, dice_smooth=1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice_smooth = float(dice_smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits = logits.float()
            target = target.float()
            probs = torch.sigmoid(logits)
            bce = self.bce(logits, target)
            inter = (probs * target).sum(dim=(1, 2, 3))
            denom = (probs + target).sum(dim=(1, 2, 3))
            dice = 1.0 - (2.0 * inter + self.dice_smooth) / (denom + self.dice_smooth)
            dice = dice.mean()
            return self.bce_weight * bce + self.dice_weight * dice
