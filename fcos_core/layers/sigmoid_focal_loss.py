import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from fcos_core import _C


# TODO: Use JIT to replace CUDA implementation in the future.
class _SigmoidFocalLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses = _C.sigmoid_focalloss_forward(
            logits, targets, num_classes, gamma, alpha
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(
            logits, targets, d_loss, num_classes, gamma, alpha
        )
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    """
    logits: (N,C), COCO: C=80
    targets: (N,)
    gamma: (1,)
    alpha: (1,)
    """
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)  # (1,C)

    t = targets.unsqueeze(1)  # (N,1), value range: 0, ..., C
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)  # (N,C)
    term2 = p ** gamma * torch.log(1 - p)  # (N,C)

    loss = -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)
    return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, weight=None, ignore_label=None, use_gpu=True):

        if ignore_label is not None:
            keep = torch.nonzero(targets!=ignore_label).view(-1)
            logits = logits[keep]
            targets = targets[keep]

        device = logits.device
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu

        loss = loss_func(logits, targets, self.gamma, self.alpha)

        if weight is None:
            return loss.sum()
        else:
            return (loss*weight).sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr


if __name__ == '__main__':
    logits = torch.tensor([[1, -1, 1, 1], [0.5, 1, 2, -3]])
    targets = torch.tensor([0, 2]).int()
    gamma = torch.tensor([2.0])
    alpha = torch.tensor([0.25])

    device = 'cuda'
    cls_loss_func = SigmoidFocalLoss(gamma, alpha)

    logits = logits.to(device)
    logits.requires_grad = True
    targets = targets.to(device)
    loss2 = cls_loss_func(logits, targets).sum()
    loss2.backward()
    print(loss2, logits.grad)