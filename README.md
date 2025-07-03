# 🧠 nnCoroUNet: 冠脉分割拓扑增强网络
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![nnUNet](https://img.shields.io/badge/nnUNet-1.7.0-3b7ab0)](https://github.com/MIC-DKFZ/nnUNet)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **基于PyTorch与nnUNet框架的冠脉CTA图像分割网络，融合拓扑感知损失与动态数据增强，提升血管连通性与边界精度。**

---

## 📖 项目概览
`nnCoroUNet` 在 `nnUNet` 基础上针对**冠脉血管分割**任务进行增强设计：
- **拓扑感知损失函数**：结合 `cl_dice_loss` 与 `auto_hd_loss`，优化血管连通性与边界准确性[2,4](@ref)
- **动态数据增强**：集成 `clahe_transform` 模块，增强低对比度血管区域特征[3](@ref)
- **中心线提取算法**：基于 `soft_skeleton` 实现3D血管骨架计算，支撑拓扑指标评估[4](@ref)

---

总目标函数为加权多损失融合，平衡分割精度与拓扑连续性：

| 损失函数               | 权重 | 功能描述                                                                 |
|------------------------|------|--------------------------------------------------------------------------|
| Cross Entropy (CE)     | 0.4  | 基础像素分类损失                                                         |
| Dice Loss              | 0.3  | 区域重叠优化                                                             |
| **clDiceLoss**[2]      | 0.2  | 基于血管骨架的拓扑精确率（防止断裂/分支丢失）                             |
| **HD Loss**[3]         | 0.1  | 边界距离约束（提升分割边缘贴合度） 

---

### 📂 文件结构
```plaintext
├── 3d_diff_visualization.html      # 交互式分割结果可视化
├── augmented_results/              # 数据增强样例
│   ├── case1_clahe.nii.gz          # CLAHE增强样本
│   ├── case2_skel.nii.gz           # 骨架提取结果
├── src/
│   ├── losses/                     
│   │   ├── cl_dice_loss.py          # [2] clDice损失
│   │   ├── auto_hd_loss.py          # [3] 动态蛇形卷积损失
│   ├── data_aug/ 
│   │   ├── clahe_transform.py       # CLAHE增强模块
│   ├── utils/
│   │   ├── soft_skeleton.py         # [4] 骨架提取算法
```
