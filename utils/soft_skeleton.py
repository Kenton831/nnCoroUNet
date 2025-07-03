# --------------------------------------------------------------- version 2 ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from scipy.ndimage import distance_transform_edt

class SoftSkeletonize_v2(nn.Module):
    def __init__(self, num_iter=20, stop_thresh=1e-4, anisotropy=(1.0, 1.0, 0.5), use_checkpoint=False):
        super().__init__()
        self.num_iter = num_iter
        self.stop_thresh = stop_thresh
        self.use_checkpoint = use_checkpoint
        self.anisotropy = nn.Parameter(torch.tensor(anisotropy))
        self._init_kernels()
        self.curr_iter = 0

    def _init_kernels(self):
        self.base_kernel = 3
        self.dilation_kernels = [
            (self.base_kernel, self.base_kernel, self.base_kernel),
            tuple(max(1, int(self.base_kernel * s)) for s in self.anisotropy)
        ]

    def _get_padding(self, kernel):
        return tuple((k - 1) // 2 for k in kernel)

    def soft_erode(self, img):
        if img.dtype == torch.bool:
            img = img.float()
        if len(img.shape) == 4:
            return self._erode_2d(img)
        elif len(img.shape) == 5:
            return self._erode_3d(img)

    def _erode_2d(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), 1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), 1, padding=(0, 1))
        return torch.min(p1, p2)

    def _erode_3d(self, img):
        pooled = []
        for k in self.dilation_kernels:
            pooled.append(-F.max_pool3d(-img, kernel_size=k, stride=1, padding=self._get_padding(k)))
        return torch.min(torch.stack(pooled), dim=0)[0]

    def soft_dilate(self, img):
        if img.dtype == torch.bool:
            img = img.float()
        if len(img.shape) == 4:
            return self._dilate_2d(img)
        elif len(img.shape) == 5:
            return self._dilate_3d(img)

    def _dilate_2d(self, img):
        dilated = []
        for k in [(3,3), (3,1), (1,3)]:
            dilated.append(F.max_pool2d(img, k, 1, padding=self._get_padding(k)))
        return torch.max(torch.stack(dilated), dim=0)[0]

    def _dilate_3d(self, img):
        pooled = []
        for k in self.dilation_kernels:
            pooled.append(F.max_pool3d(img, kernel_size=k, stride=1, padding=self._get_padding(k)))
        return torch.max(torch.stack(pooled), dim=0)[0]

    def soft_open(self, img):
        if self.use_checkpoint and self.training:
            return checkpoint(self._open_impl, img)
        return self._open_impl(img)

    def _open_impl(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img = img.clone()
        skel = F.leaky_relu(img - self.soft_open(img), 0.01)
        prev_skel = torch.zeros_like(skel)
        for i in range(self.num_iter):
            self.curr_iter = i
            img = self._erode_with_checkpoint(img)
            lr = 0.1 * (0.5 ** (i // (self.num_iter // 5)))
            img_open = self.soft_open(img)
            delta = F.leaky_relu(img - img_open, 0.01)
            skel_update = F.leaky_relu(delta - skel*delta, 0.01) * lr
            skel = skel + skel_update
            delta_norm = (skel - prev_skel).abs().mean()
            if delta_norm < self.stop_thresh:
                break
            prev_skel = skel.clone()
        return skel

    def _erode_with_checkpoint(self, img):
        if self.use_checkpoint and (self.curr_iter % 3 == 0):
            return checkpoint(self.soft_erode, img)
        return self.soft_erode(img)

    def forward(self, img):
        if self.use_checkpoint and self.training:  # 仅在训练时启用
            return checkpoint(self._forward_impl, img)
        return self._forward_impl(img)

    def _forward_impl(self, img):
        skel = self.soft_skel(img)
        dt = self.distance_transform(skel.detach())
        return skel * (1 + 0.2 * torch.sigmoid(dt))

    # def distance_transform(self, x):
    #     dt = torch.zeros_like(x)
    #     B, C, *spatial_dims = x.shape
    #     for b in range(B):
    #         for c in range(C):
    #             mask = x[b, c] > 0
    #             if mask.sum() == 0:
    #                 dt[b, c] = 0
    #                 continue
    #             # 使用 scipy 的欧氏距离变换（需安装 scipy）
    #             from scipy.ndimage import distance_transform_edt
    #             dt[b, c] = torch.from_numpy(
    #                 distance_transform_edt(mask.cpu().numpy())
    #             ).to(x.device)
    #     return dt
    # def distance_transform(self, x):
    #     # 向量化实现 (假设输入是二值化的)
    #     dt = torch.zeros_like(x)
    #     for b in range(x.shape[0]):
    #         # 使用PyTorch内置操作 (需要转换为二进制掩码)
    #         mask = (x[b] > 0).cpu().numpy()
    #         dt_np = distance_transform_edt(mask)
    #         dt[b] = torch.from_numpy(dt_np).to(x.device)
    #     return dt
    def distance_transform(self, x):
        # 二值化
        binary = (x > 0).float()

        # 计算距离变换（近似，用 PyTorch 的卷积操作替代 SciPy，完全在 GPU 上运行）
        dt = -1 * (1 - binary) * (F.max_pool3d(-binary, kernel_size=3, stride=1, padding=1))
        return dt
    @torch.no_grad()
    def visualize_anisotropy(self):
        import matplotlib.pyplot as plt
        kernel = self.dilation_kernels[1].cpu().numpy()
        plt.figure(figsize=(10,5))
        plt.subplot(121).imshow(np.zeros((7,7)), vmin=0, vmax=1)
        plt.title(f'Base Kernel: 3x3x3')
        plt.subplot(122).imshow(np.zeros((kernel[0], kernel[1])))
        plt.title(f'Aniso Kernel: {kernel[0]}x{kernel[1]}x{kernel[2]}')
        plt.show()
# --------------------------------------------------------------- version 1 ------------------------------------------------------------------------------

class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter=20, stop_thresh=1e-4, anisotropy=(1,1,0.5)):
        super().__init__()
        self.num_iter = num_iter
        self.stop_thresh = stop_thresh
        self.anisotropy = anisotropy  # (xy, z) scaling factors
        
        # 使用以下参数替代原有设置
        self.kernel_size = 3  # 固定kernel大小，必须为奇数
        self.padding = self.kernel_size // 2  # 自动计算padding
        self.stride = 1

    def soft_erode(self, img):
        if img.dtype == torch.bool:
            img = img.float()

        if len(img.shape) == 4:  # 2D数据
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        
        elif len(img.shape) == 5:  # 3D数据
            # 统一使用固定参数
            kernel = (self.kernel_size, self.kernel_size, self.kernel_size)
            padding = (self.padding, self.padding, self.padding)
            # 保持各向异性处理但适配新参数
            kernels = [
                kernel,  # 原XY平面处理
                (max(1, int(self.kernel_size*self.anisotropy[0])),
                 max(1, int(self.kernel_size*self.anisotropy[1])),
                 max(1, int(self.kernel_size*self.anisotropy[2])))
            ]
            
            pooled = []
            for k in kernels:
                # 自动对齐padding
                actual_padding = (
                    max(0, (k[0]-1)//2),
                    max(0, (k[1]-1)//2),
                    max(0, (k[2]-1)//2)
                )
                pooled.append(
                    -F.max_pool3d(
                        -img, 
                        kernel_size=k,
                        stride=self.stride,
                        padding=actual_padding
                    )
                )
            
            # 尺寸对齐保障（可选）
            target_size = pooled[0].shape[2:]
            aligned = []
            for p in pooled:
                if p.shape[2:] != target_size:
                    p = F.interpolate(p, size=target_size, mode='nearest')
                aligned.append(p)
                
            return torch.min(torch.stack(aligned), dim=0)[0]

    def soft_dilate(self, img):
        if img.dtype == torch.bool:
            img = img.float()

        if len(img.shape) == 4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape) == 5:
            # 根据方案二修改的3D膨胀
            return F.max_pool3d(
                img, 
                kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                stride=self.stride,
                padding=(self.padding, self.padding, self.padding)
            )

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img = img.clone()
        skel = F.leaky_relu(img - self.soft_open(img), negative_slope=0.01)
        
        prev_skel = torch.zeros_like(skel)
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img_open = self.soft_open(img)
            delta = F.leaky_relu(img - img_open, negative_slope=0.01)
            
            # 自适应融合
            skel_update = F.leaky_relu(delta - skel*delta, 0.01)
            skel = skel + skel_update
            
            # 动态停止条件
            delta_norm = (skel - prev_skel).abs().mean()
            if delta_norm < self.stop_thresh:
                break
            prev_skel = skel.clone()

        return skel

    def forward(self, img):
        # 使用梯度检查点节省显存
        return checkpoint(self._forward_impl, img)
    
    def _forward_impl(self, img):
        skel = self.soft_skel(img)
        
        # 后处理增强（可选，需要距离变换支持）
        if hasattr(self, 'distance_transform'):
            with torch.no_grad():
                dt = self.distance_transform(img)
            return skel * (1 + 0.2*torch.sigmoid(dt))
        return skel

    @staticmethod
    def distance_transform(prob_map):
        # 需要自定义或使用第三方实现的3D距离变换
        # 此处为占位符实现
        return torch.zeros_like(prob_map)


# --------------------------------------------------------------- former version ------------------------------------------------------------------------------


# class SoftSkeletonize(torch.nn.Module):

#     def __init__(self, num_iter=40):

#         super(SoftSkeletonize, self).__init__()
#         self.num_iter = num_iter

#     def soft_erode(self, img):
#         if img.dtype == torch.bool:
#             img = img.float()

#         if len(img.shape) == 4:
#             p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
#             p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
#             return torch.min(p1, p2)
#         elif len(img.shape) == 5:
#             p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
#             p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
#             p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))  
#             return torch.min(p3, torch.min(p1, p2))  

#     def soft_dilate(self, img):
#         if img.dtype == torch.bool:
#             img = img.float()

#         if len(img.shape) == 4:
#             return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
#         elif len(img.shape) == 5:
#             return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))   # 这里也降低了一下

#     def soft_open(self, img):

#         return self.soft_dilate(self.soft_erode(img))

#     def soft_skel(self, img):
#         if img.dtype == torch.bool:
#             img = img.float()

#         img1 = self.soft_open(img)
#         skel = F.relu(img - img1)

#         for j in range(self.num_iter):
#             img = self.soft_erode(img)
#             img1 = self.soft_open(img)
#             delta = F.relu(img - img1)
#             skel = skel + F.relu(delta - skel * delta)

#         return skel

#     def forward(self, img):

#         return self.soft_skel(img)
