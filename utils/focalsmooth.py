import torch
import torch.nn as nn

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=1, gamma=2, epsilon=0.1, reduction='mean'):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 应用 Label Smoothing
        targets_smoothed = targets.float() * (1 - self.epsilon) + self.epsilon * 0.5
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets_smoothed, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss.sum()
