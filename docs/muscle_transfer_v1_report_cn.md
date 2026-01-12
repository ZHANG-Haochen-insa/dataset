# 肌肉迁移学习 V1 方法报告

## 概述

本报告记录了肌肉迁移学习V1版本的设计思路、实现方法和遇到的问题。V1是第一次尝试使用迁移学习方法来分割全身肌肉。

## 核心思想

V1采用了**Teacher-Student架构**，核心理念是：

```
已有：训练好的脊椎附近肌肉分割模型（Teacher）
目标：训练一个能分割全身所有肌肉的新模型（Student）
方法：让Student在已知区域与Teacher保持一致，在未知区域根据学到的特征自行判断
```

### 设计理念

1. **从局部到整体**：已有脊椎附近肌肉的分割模型，希望扩展到全身
2. **特征迁移**：让模型学习"肌肉的特征"（HU值分布、纹理等）
3. **自监督扩展**：在未标注区域，模型根据学到的特征自行分割

## 网络架构

### Student模型：MuscleTransferNet

```python
class MuscleTransferNet(nn.Module):
    """
    肌肉特征迁移网络

    输入: 3通道
        - CT图像 (1通道)
        - HU特征图 (1通道，表示是否在肌肉HU范围内)
        - Teacher预测 (1通道，已知肌肉区域)
    输出:
        - 全身肌肉分割 (1通道)
        - 肌肉特征向量 (32维，用于后续分析)
    """
```

**架构特点**：

| 组件 | 说明 |
|------|------|
| 编码器 | 4层下采样，特征通道 [32, 64, 128, 256] |
| 注意力模块 | AttentionBlock，帮助模型关注肌肉特征 |
| 解码器 | 4层上采样，带skip连接 |
| 肌肉特征编码器 | 从瓶颈层提取32维特征向量 |
| Dropout | 编码器0.1，瓶颈层0.2 |

### Teacher模型：UNet2D

```python
class UNet2D(nn.Module):
    """Teacher模型结构（与原训练一致）"""
    # 标准UNet结构，无Dropout
    # 输入: 1通道CT
    # 输出: 多通道肌肉分割
```

## 损失函数设计

```python
class MuscleTransferLoss(nn.Module):
    """
    四项损失组成：
    1. consistency - Teacher一致性损失
    2. hu_prior - HU先验损失
    3. coverage - 覆盖率损失
    4. boundary - 边界平滑损失
    """
```

### 损失权重配置

| 损失项 | 权重 | 作用 |
|--------|------|------|
| consistency | 2.0 | 在Teacher预测区域，必须与Teacher一致 |
| hu_prior | 1.0 | 预测为肌肉的区域，HU值应该在范围内 |
| coverage | 1.0 | 不能丢失Teacher预测的肌肉 |
| boundary | 0.5 | 边界平滑性约束 |

### 损失函数详解

```python
def forward(self, pred_logits, teacher_mask, hu_slice, body_mask):
    # 1. 一致性损失 - Teacher区域权重更高
    weight_map = 1 + teacher_region * 2.0  # Teacher区域权重3倍
    consistency_loss = (bce_loss * weight_map * body_mask).sum() / body_mask.sum()

    # 2. HU先验损失 - 惩罚预测为肌肉但HU不在范围内的区域
    hu_violation = pred_prob * (1 - hu_in_range) * body_mask
    hu_prior_loss = hu_violation.mean()

    # 3. 覆盖率损失 - 确保覆盖已知肌肉
    coverage = (pred_prob * teacher_region).sum() / teacher_region.sum()
    coverage_loss = 1 - coverage

    # 4. 边界平滑损失 - 总变差(TV)损失
    tv_h = |pred[:,:,1:,:] - pred[:,:,:-1,:]|
    tv_w = |pred[:,:,:,1:] - pred[:,:,:,:-1]|
    boundary_loss = tv_h + tv_w
```

## 数据集设计

### 输入构成（3通道）

```python
input_tensor = np.stack([
    ct_resized,           # 通道0: 归一化CT图像
    hu_muscle_feature,    # 通道1: HU特征图（是否在肌肉范围内）
    teacher_pred,         # 通道2: Teacher模型预测
], axis=0)
```

### Teacher预测获取

```python
def _get_teacher_prediction(self, ct_normalized, original_shape):
    """获取Teacher模型预测"""
    img_t = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = torch.sigmoid(self.teacher_model(img_t))
        # 合并所有肌肉类别
        pred_combined = (pred[0].sum(dim=0) > 0.5).float()

    return pred_combined
```

## 评估指标

```python
def compute_metrics(pred, teacher, hu_slice, body_mask):
    """
    五个评估指标：
    1. teacher_coverage - Teacher区域覆盖率
    2. teacher_dice - 与Teacher预测的Dice分数
    3. hu_compliance - HU值合规率
    4. expansion_ratio - 相对Teacher的扩展比例
    5. overflow_rate - 溢出到非肌肉HU区域的比例
    """
```

## HU值范围定义

```python
# 肌肉HU值范围
MUSCLE_HU_MIN = -29
MUSCLE_HU_MAX = 150
MUSCLE_HU_OPTIMAL_MIN = 0   # 典型肌肉
MUSCLE_HU_OPTIMAL_MAX = 100

# 其他组织
BONE_HU_MIN = 200   # 骨骼
FAT_HU_MIN = -190   # 脂肪
FAT_HU_MAX = -30
AIR_HU_MAX = -500   # 空气
```

## 训练配置

```python
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
EPOCHS = 30

# 可视化配置
VIS_EVERY_N_EPOCHS = 2
VIS_NUM_SAMPLES = 4
```

## 遇到的问题

### 1. 模型结构不匹配（致命问题）

**问题描述**：
尝试加载预训练的Teacher模型时，遇到权重不匹配错误：

```
RuntimeError: Error(s) in loading state_dict for UNet2D:
    Unexpected key(s) in state_dict: "encoder.0.net.3.weight"...
```

**原因分析**：
原始训练的模型使用了带Dropout的`DoubleConv`：

```python
# 原始模型结构
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # 这里有Dropout！
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
```

而V1代码中定义的Teacher模型没有Dropout层，导致层索引不匹配。

### 2. 依赖模型预测作为伪标签

**问题**：使用Teacher模型的预测作为监督信号，可能会传播错误。

### 3. 缺乏硬排除约束

**问题**：对空气和骨骼区域没有足够强的排除机制，可能导致假阳性。

### 4. 架构复杂度高

**问题**：
- 需要同时维护Teacher和Student两个模型
- 数据集构建时需要运行Teacher推理，增加了处理时间
- 调试难度增加

## 改进方向（导致V2）

1. **移除Teacher模型依赖**：直接使用TotalSegmentator的真实分割标签
2. **增加硬约束**：对空气和骨骼区域增加强惩罚
3. **简化架构**：只训练一个Student模型
4. **添加区域筛选**：专注于腹部区域，排除大腿

## 文件信息

| 属性 | 值 |
|------|-----|
| 文件路径 | `scripts/train_muscle_transfer.py` |
| 创建日期 | 2026-01-10 |
| 代码行数 | ~700行 |
| 状态 | 已被V2替代 |

## 总结

V1版本提出了一个创新的Teacher-Student迁移学习框架，但由于模型结构不匹配的问题未能成功运行。这个问题促使了V2版本的设计，V2直接使用TotalSegmentator标签而非模型预测，简化了整个流程。

尽管V1未能成功运行，但它的设计思想（从局部到整体、特征迁移、自监督扩展）为后续版本奠定了基础。

---

*报告生成时间: 2026-01-11*
