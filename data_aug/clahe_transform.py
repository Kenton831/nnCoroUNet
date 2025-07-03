import numpy as np
import cv2
import torch

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), apply_probability=1.0):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.apply_probability = apply_probability  # 变换应用概率

    def __call__(self, **kwargs):
        # 兼容 batchgenerators，**kwargs 可能直接传入 'image' 或 'data'
        data_dict = kwargs

        # 确保 'data' 或 'image' 键存在
        images = data_dict.get('data', data_dict.get('image', None))
        if images is None:
            raise KeyError("Neither 'data' nor 'image' found in data_dict.")

        # 处理图像数据
        images = np.array(images)  # 确保转换为 numpy 数组
        if np.random.rand() < self.apply_probability:  # 按概率决定是否应用 CLAHE
            if len(images.shape) == 4:  # (B, C, H, W) 格式
                for b in range(images.shape[0]):
                    for c in range(images.shape[1]):
                        images[b, c] = self.apply_clahe(images[b, c])
            elif len(images.shape) == 3:  # (C, H, W) 格式
                for c in range(images.shape[0]):
                    images[c] = self.apply_clahe(images[c])
            else:
                raise ValueError("Unexpected image shape: {}".format(images.shape))

        # 如果数据已经是 Tensor 格式，则直接返回
        if isinstance(images, torch.Tensor):
            images = images.float()  # 保证为 float 类型

        # 更新 data_dict
        if 'data' in data_dict:
            data_dict['data'] = torch.tensor(images)  # 转换为 Tensor
        else:
            data_dict['image'] = torch.tensor(images)  # 转换为 Tensor

        return data_dict

    def apply_clahe(self, image):
        image = np.clip(image, 0, 255).astype(np.uint8)  # CLAHE 需要 uint8 输入
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(image)
