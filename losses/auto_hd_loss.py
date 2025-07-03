import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------version 2 -----------------------------------------------------
class AutoHausdorffDTLoss(nn.Module):
    def __init__(self, dim='auto', alpha=1.5, kernel_size=5, voxel_spacing=None):
        super().__init__()
        self.dim_mode = dim
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.voxel_spacing = voxel_spacing or (1.0, 1.0, 1.0)
        self.kernel_cache = {}

        self.dim_config = {
            2: {
                'kernel_generator': self._create_2d_kernel,
                'conv_fn': F.conv2d,
                'dt_scale': 1.0
            },
            3: {
                'kernel_generator': self._create_3d_kernel,
                'conv_fn': F.conv3d,
                'dt_scale': 1.2
            }
        }

    def _create_2d_kernel(self, spacing):
        key = (2, tuple(spacing), self.kernel_size, self.alpha)
        if key not in self.kernel_cache:
            x = torch.linspace(-1, 1, self.kernel_size) * spacing[0]
            y = torch.linspace(-1, 1, self.kernel_size) * spacing[1]
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            kernel = -(xx**2 + yy**2) * self.alpha
            self.kernel_cache[key] = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        return self.kernel_cache[key]

    def _create_3d_kernel(self, spacing):
        key = (3, tuple(spacing), self.kernel_size, self.alpha)
        if key not in self.kernel_cache:
            z = torch.linspace(-1, 1, self.kernel_size) * spacing[0]
            x = torch.linspace(-1, 1, self.kernel_size) * spacing[1]
            y = torch.linspace(-1, 1, self.kernel_size) * spacing[2]
            zz, xx, yy = torch.meshgrid(z, x, y, indexing='ij')
            kernel = -(zz**2 + xx**2 + yy**2) * self.alpha
            self.kernel_cache[key] = kernel.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)
        return self.kernel_cache[key]

    def _get_dimension(self, x):
        return 3 if x.dim() == 4 else 2  # 4D输入→3D（B, D, H, W），3D输入→2D（B, H, W）

    def _compute_dt(self, mask):
        dim = self._get_dimension(mask)
        cfg = self.dim_config[dim]

        # 生成卷积核
        kernel = cfg['kernel_generator'](self.voxel_spacing[:dim]).to(mask.device)

        # 添加通道维度
        mask = mask.unsqueeze(1)  # 3D → [B,1,D,H,W]，2D → [B,1,H,W]

        # 动态计算填充量（每个维度左右各填充 kernel_size//2）
        if dim == 2:
            pad = (self.kernel_size//2, self.kernel_size//2,   # 左右填充
                   self.kernel_size//2, self.kernel_size//2)  # 上下填充
        else:
            pad = (self.kernel_size//2, self.kernel_size//2,   # 左右填充
                   self.kernel_size//2, self.kernel_size//2,  # 前后填充
                   self.kernel_size//2, self.kernel_size//2)   # 上下填充

        # 填充操作
        padded_mask = 1 - F.pad(mask, pad, mode='constant', value=0)

        # 调用卷积时不需要额外padding（已在F.pad中完成）
        dt = cfg['conv_fn'](padded_mask, kernel, padding=0).clamp(min=0) * cfg['dt_scale']

        # 打印调试信息
        # print(f"Input mask shape: {mask.shape}")
        # print(f"Padded mask shape: {padded_mask.shape}")
        # print(f"Kernel shape: {kernel.shape}")
        # print(f"Output shape after conv: {dt.shape}")

        return dt.squeeze(1)

    def forward(self, pred_logits, target):
        # 处理 target 的维度
        if target.dim() == 5 and target.shape[1] > 1:
            target_mask = target.argmax(dim=1)  # [B, D, H, W]
        else:
            target_mask = target.squeeze(1)     # [B, H, W]
        target_mask = (target_mask > 0).float()

        # 处理 pred_logits 的维度
        pred_mask = torch.sigmoid(pred_logits)
        if pred_mask.dim() == 5 and pred_mask.shape[1] > 1:
            pred_mask = pred_mask.argmax(dim=1)  # [B, D, H, W]
        else:
            pred_mask = pred_mask.squeeze(1)     # [B, H, W]
        pred_mask = (pred_mask > 0).float()

        # 计算距离变换
        dt_target = self._compute_dt(target_mask)
        dt_pred = self._compute_dt(1 - pred_mask)

        # 计算 HD 损失
        hd_s2t = (torch.sigmoid(pred_logits) * dt_target).flatten(1).max(dim=1)[0]
        hd_t2s = (target_mask * dt_pred).flatten(1).max(dim=1)[0]
        return (hd_s2t + hd_t2s).mean()
# ----------------------------------------------------------------- version 1 -----------------------------------------------------
# class AutoHausdorffDTLoss(nn.Module):
#     def __init__(self, dim='auto', alpha=1.5, kernel_size=5, voxel_spacing=None):
#         super().__init__()
#         self.dim_mode = dim
#         self.alpha = alpha
#         self.kernel_size = kernel_size
#         self.voxel_spacing = voxel_spacing or (1.0, 1.0, 1.0)
        
#         # 预定义不同维度的配置
#         self.dim_config = {
#             2: {
#                 'kernel_generator': self._create_2d_kernel,
#                 'padding': (1,1,1,1),
#                 'conv_fn': F.conv2d,
#                 'dt_scale': 1.0
#             },
#             3: {
#                 'kernel_generator': self._create_3d_kernel,
#                 'padding': (1,1,1,1,1,1),
#                 'conv_fn': F.conv3d,
#                 'dt_scale': 1.2
#             }
#         }

#     def _create_2d_kernel(self, spacing):
#         x = torch.linspace(-1, 1, self.kernel_size) * spacing[0]
#         y = torch.linspace(-1, 1, self.kernel_size) * spacing[1]
#         xx, yy = torch.meshgrid(x, y, indexing='ij')
#         kernel = -(xx**2 + yy**2) * self.alpha
#         return kernel.view(1, 1, self.kernel_size, self.kernel_size)

#     def _create_3d_kernel(self, spacing):
#         z = torch.linspace(-1, 1, self.kernel_size) * spacing[0]
#         x = torch.linspace(-1, 1, self.kernel_size) * spacing[1]
#         y = torch.linspace(-1, 1, self.kernel_size) * spacing[2]
#         zz, xx, yy = torch.meshgrid(z, x, y, indexing='ij')
#         kernel = -(zz**2 + xx**2 + yy**2) * self.alpha
#         return kernel.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)

#     def _get_dimension(self, x):
#         if self.dim_mode == 'auto':
#             return x.dim() - 2  # 排除batch和channel维度
#         return self.dim_mode

#     def _compute_dt(self, mask):
#         dim = self._get_dimension(mask)
#         cfg = self.dim_config[dim]
        
#         # 生成动态卷积核
#         kernel = cfg['kernel_generator'](self.voxel_spacing[:dim]).to(mask.device)
        
#         # 对称填充
#         padded_mask = 1 - F.pad(mask, cfg['padding'], mode='replicate')
        
#         # 执行卷积
#         dt = cfg['conv_fn'](padded_mask, kernel, padding=0)
        
#         return dt.clamp(min=0) * cfg['dt_scale']

#     def forward(self, pred_logits, target):
#         # 自动维度检测
#         dim = self._get_dimension(pred_logits)
#         assert dim in [2,3], f"Unsupported dimension: {dim}, must be 2 or 3"
        
#         # 概率转换
#         pred_probs = torch.sigmoid(pred_logits)
        
#         # 二值化目标
#         target_mask = (target > 0).float()
        
#         # 双向距离变换
#         dt_target = self._compute_dt(target_mask)
#         dt_pred = self._compute_dt(1 - pred_probs)
        
#         # Hausdorff距离计算
#         hd_s2t = (pred_probs * dt_target).flatten(1).max(dim=1)[0]
#         hd_t2s = (target_mask * dt_pred).flatten(1).max(dim=1)[0]
        
#         return (hd_s2t + hd_t2s).mean()
