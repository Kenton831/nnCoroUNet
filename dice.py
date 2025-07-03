from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
from .soft_skeleton import SoftSkeletonize


class MemoryEfficientSoftCLDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        super(MemoryEfficientSoftCLDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))  # 确定空间维度，如3D时为(2,3,4)

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y.float()  # 确保为浮点
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)
                y_onehot = y_onehot.float()  # 转换为浮点

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:]
        x = x.float()

        skel_pred = self.soft_skeletonize(x)
        skel_true = self.soft_skeletonize(y_onehot)

        # 处理loss_mask的广播问题
        if loss_mask is not None:
            if loss_mask.dim() == skel_pred.dim() - 1:
                loss_mask = loss_mask.unsqueeze(1)
            loss_mask_broadcast = loss_mask.expand_as(skel_pred)
        else:
            loss_mask_broadcast = 1.0

        # 计算tprec（拓扑精确率）
        tprec_numerator = torch.sum(skel_pred * y_onehot * loss_mask_broadcast, dim=axes)
        tprec_denominator = torch.sum(skel_pred * loss_mask_broadcast, dim=axes)
        tprec = (tprec_numerator + self.smooth) / (tprec_denominator + self.smooth)

        # 计算tsens（拓扑敏感率）
        tsens_numerator = torch.sum(skel_true * x * loss_mask_broadcast, dim=axes)
        tsens_denominator = torch.sum(skel_true * loss_mask_broadcast, dim=axes)
        tsens = (tsens_numerator + self.smooth) / (tsens_denominator + self.smooth)

        # 计算各通道的交集
        intersect = tprec * tsens

        # 分布式训练处理
        if self.batch_dice and self.ddp:
            intersect = AllGatherGrad.apply(intersect).sum(0)
            tprec_total = AllGatherGrad.apply(tprec).sum(0)
            tsens_total = AllGatherGrad.apply(tsens).sum(0)
        elif self.batch_dice:
            intersect = intersect.sum(0)
            tprec_total = tprec.sum(0)
            tsens_total = tsens.sum(0)
        else:
            tprec_total = tprec
            tsens_total = tsens

        # 计算clDice系数
        cl_dc = (2 * intersect + self.smooth) / (tprec_total + tsens_total + self.smooth)
        cl_dc = cl_dc.mean()

        return -cl_dc


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl = MemoryEfficientSoftCLDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0,
                                         ddp=False)
    res = dl(pred, ref)
    print(res)
