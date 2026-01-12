# 肌肉迁移学习方法演进报告

## 概述

本报告记录了肌肉迁移学习方法从V1到V3的完整演进过程，包括每个版本的设计思路、实现细节、遇到的问题以及改进方向。

**核心目标**：利用已标注的脊椎附近肌肉（autochthon、gluteus、iliopsoas）学习肌肉特征，进而分割全身所有肌肉组织，尤其是未被标注的腹部肌肉。

---

## 第一阶段：V1 - Teacher-Student架构

### 设计理念

V1采用了**Teacher-Student架构**，核心思想是：

```
已有：训练好的脊椎附近肌肉分割模型（Teacher）
目标：训练一个能分割全身所有肌肉的新模型（Student）
方法：Student在已知区域与Teacher保持一致，在未知区域根据学到的特征自行判断
```

### 网络架构

**Student模型：MuscleTransferNet**

```python
class MuscleTransferNet(nn.Module):
    """
    输入: 3通道
        - CT图像
        - HU特征图（是否在肌肉HU范围内）
        - Teacher预测（已知肌肉区域）
    输出:
        - 全身肌肉分割
        - 肌肉特征向量（32维）
    """
```

| 组件 | 说明 |
|------|------|
| 编码器 | 4层下采样，特征通道 [32, 64, 128, 256] |
| 注意力模块 | AttentionBlock，帮助模型关注肌肉特征 |
| 解码器 | 4层上采样，带skip连接 |
| 肌肉特征编码器 | 从瓶颈层提取32维特征向量 |

### 损失函数

```python
class MuscleTransferLoss:
    # 四项损失
    consistency = 2.0   # Teacher一致性
    hu_prior = 1.0      # HU先验约束
    coverage = 1.0      # 覆盖率（不丢失已知肌肉）
    boundary = 0.5      # 边界平滑
```

### 遇到的问题

**致命问题：模型结构不匹配**

```
RuntimeError: Error(s) in loading state_dict for UNet2D:
    Unexpected key(s) in state_dict: "encoder.0.net.3.weight"...
```

原因：原始训练的模型使用了带Dropout的`DoubleConv`，而V1代码中定义的Teacher模型没有Dropout层，导致层索引不匹配。

```python
# 原始模型（有Dropout）
nn.Sequential(
    nn.Conv2d(...),        # index 0
    nn.BatchNorm2d(...),   # index 1
    nn.ReLU(...),          # index 2
    nn.Dropout2d(...),     # index 3  ← 多了这一层！
    nn.Conv2d(...),        # index 4
    ...
)

# V1定义的Teacher（无Dropout）
nn.Sequential(
    nn.Conv2d(...),        # index 0
    nn.BatchNorm2d(...),   # index 1
    nn.ReLU(...),          # index 2
    nn.Conv2d(...),        # index 3  ← 索引不匹配
    ...
)
```

### V1总结

| 项目 | 状态 |
|------|------|
| 创新点 | Teacher-Student架构、注意力机制、特征向量提取 |
| 问题 | 模型结构不匹配，无法加载预训练权重 |
| 结果 | 未能成功运行 |

---

## 第二阶段：V2 - 直接使用TotalSegmentator标签

### 设计改进

针对V1的问题，V2做出了根本性改变：

```
V1: 使用Teacher模型预测作为伪标签 → 需要加载预训练模型 → 结构不匹配
V2: 直接使用TotalSegmentator的真实分割标签 → 不依赖任何预训练模型
```

### 核心改动

**1. 移除Teacher模型依赖**

```python
# V2 - 直接读取TotalSegmentator分割结果
KNOWN_MUSCLE_FILES = [
    'autochthon_left.nii.gz',
    'autochthon_right.nii.gz',
    'gluteus_maximus_left.nii.gz',
    'gluteus_maximus_right.nii.gz',
    'gluteus_medius_left.nii.gz',
    'gluteus_medius_right.nii.gz',
    'gluteus_minimus_left.nii.gz',
    'gluteus_minimus_right.nii.gz',
    'iliopsoas_left.nii.gz',
    'iliopsoas_right.nii.gz',
]
```

**2. 增强HU约束**

```python
# 硬排除区域定义
AIR_HU_MAX = -200      # 空气，必须排除
BONE_HU_MIN = 300      # 骨骼，必须排除

# 损失权重
LABEL_WEIGHT = 2.0            # 已标注区域一致性
HU_CONSTRAINT_WEIGHT = 3.0    # HU约束
EXCLUSION_WEIGHT = 5.0        # 排除区域（最强！）
BOUNDARY_WEIGHT = 0.3         # 边界平滑
```

**3. 新的损失函数**

```python
class MuscleTransferLossV2:
    def forward(self, pred_logits, label_mask, hu_slice, body_mask, exclusion_mask):
        # 1. 标签一致性损失
        label_loss = BCE(pred, label) * weight

        # 2. 排除区域损失（最重要！空气/骨骼必须为0）
        exclusion_loss = (pred_prob * exclusion_mask).mean()

        # 3. HU约束损失（HU无效区域不能预测为肌肉）
        hu_loss = (pred_prob * hu_invalid * body_mask).mean() * 10

        # 4. 边界平滑损失
        boundary_loss = TV(pred_prob)
```

**4. 腹部区域筛选**

用户反馈希望专注于腹部区域，排除大腿部分：

```python
# 腹部器官标记
ABDOMINAL_ORGANS = ['liver', 'spleen', 'kidney_left', 'kidney_right', ...]

# 排除标记（大腿）
EXCLUDE_MARKERS = ['femur_left', 'femur_right']

# 筛选逻辑
keep_slice = has_abdominal_organ AND NOT has_femur
```

筛选效果：
- 训练集：6352/7695 切片保留（82.5%）
- 验证集：2119/2540 切片保留（83.4%）

**5. 简化的输入设计（2通道）**

```python
input_tensor = np.stack([
    ct_resized,       # 通道0: CT图像
    muscle_resized,   # 通道1: 已知肌肉标签
], axis=0)
```

### 训练结果

| 指标 | 数值 |
|------|------|
| HU符合率 | 99.5% |
| Dice分数 | 0.80 |
| 排除损失 | 接近0 |

### 发现的问题

**模型过于保守**：尽管指标不错，但实际分析发现预测的肌肉区域覆盖面积不足，即使是已标注的肌肉区域也没有被完全覆盖。

用户反馈：
> "我希望它能够覆盖更多的肌肉，而不是现在甚至已经在样本库中的肌肉量都没有被完全覆盖住。"

### V2总结

| 项目 | 状态 |
|------|------|
| 改进点 | 移除Teacher依赖、增强HU约束、腹部筛选 |
| 成功点 | 训练成功，指标良好 |
| 问题 | 模型过于保守，覆盖率不足 |

---

## 第三阶段：V3 - "先扩后缩"方法

### 设计理念

针对V2覆盖率不足的问题，用户提出了"先扩后缩"的思路：

```
先扩：覆盖所有HU值在肌肉范围内的像素（高召回率）
后缩：通过与已知标签对齐来精确边界（高精确率）
```

用户原话：
> "你可以先像之前那样定义一个HU的值，然后覆盖这个CT图上的所有的在这个值内的点。然后再进行一个小幅度的缩减，让它达到在分割肌肉那块，就是已经有样本的分割肌肉那块，进行一个符合的描边。"

### 核心改动

**1. 扩展的输入设计（4通道）**

```python
input_tensor = np.stack([
    ct_resized,           # 通道0: CT图像
    hu_coarse,            # 通道1: HU粗分割（所有HU在-29~150的像素）← 新增！
    muscle_resized,       # 通道2: 已知肌肉标签
    1 - exclusion         # 通道3: 非排除区域掩码 ← 新增！
], axis=0)
```

**设计原因**：
- `hu_coarse`告诉模型"这些像素在物理上可能是肌肉"，鼓励扩大覆盖
- `1 - exclusion`明确标出不可能是肌肉的区域

**2. 新的损失函数：ExpandThenRefineLoss**

```python
class ExpandThenRefineLoss(nn.Module):
    """
    三项损失，实现"先扩后缩"：
    1. label_alignment (权重3.0) - 已标注区域精确对齐
    2. coverage_reward (权重1.0) - 鼓励覆盖HU有效的未标注区域
    3. exclusion (权重5.0) - 严格排除空气/骨骼
    """

    def forward(self, pred_logits, label_mask, hu_coarse, exclusion_mask, body_mask):
        # 1. 已标注区域对齐损失 - 确保边界精度
        has_label = (label_mask > 0.5).float()
        label_region_loss = (BCE * has_label).sum() / has_label.sum()
        losses['label_alignment'] = label_region_loss * 3.0

        # 2. 覆盖奖励 - 鼓励覆盖未标注但HU有效的区域（关键！）
        unlabeled_hu_valid = hu_coarse * (1 - has_label) * body_mask * (1 - exclusion_mask)
        coverage_loss = ((1 - pred_prob) * unlabeled_hu_valid).mean()
        losses['coverage_reward'] = coverage_loss * 1.0

        # 3. 排除区域损失 - 绝对禁止预测空气/骨骼
        exclusion_loss = (pred_prob * exclusion_mask).mean()
        losses['exclusion'] = exclusion_loss * 5.0
```

**损失函数对比**：

| 损失项 | V2 | V3 |
|--------|----|----|
| 标签对齐 | BCE on all | BCE only on labeled regions (weight 3.0) |
| HU约束 | 惩罚HU无效区域 | 奖励覆盖HU有效区域 |
| 排除 | 排除空气/骨骼 | 排除空气/骨骼（相同） |
| 边界平滑 | TV损失 | 移除（依赖标签对齐） |

### V3配置

```python
# 超参数
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
IMAGE_SIZE = 256

# HU范围
HU_MIN = -29
HU_MAX = 150

# 排除阈值
AIR_THRESHOLD = -200
BONE_THRESHOLD = 300

# 损失权重
LABEL_ALIGNMENT_WEIGHT = 3.0   # 标签对齐（精度）
COVERAGE_REWARD_WEIGHT = 1.0   # 覆盖奖励（召回率）
EXCLUSION_WEIGHT = 5.0         # 排除约束
```

### 训练状态

- **启动时间**: 2026-01-11 01:08
- **进程PID**: 2333264
- **训练数据**: 6,352个腹部切片
- **验证数据**: 2,119个腹部切片
- **WandB**: https://wandb.ai/haochen-zhang-insa-lyon/muscle-transfer-learning

### 预期效果

1. **更高的召回率**：覆盖更多肌肉区域
2. **保持边界精度**：通过label_alignment损失
3. **避免假阳性**：通过exclusion损失

---

## 版本演进总结

```
V1: Teacher-Student架构
    ├─ 创新: 注意力机制、特征向量提取
    ├─ 问题: 模型结构不匹配
    └─ 状态: 失败

         ↓ 移除Teacher依赖，使用真实标签

V2: 直接使用TotalSegmentator标签
    ├─ 改进: 硬HU约束、腹部筛选
    ├─ 结果: HU符合率99.5%, Dice 0.80
    ├─ 问题: 模型过于保守，覆盖率不足
    └─ 状态: 成功但不满意

         ↓ 增加覆盖奖励，"先扩后缩"

V3: "先扩后缩"方法
    ├─ 改进: 4通道输入、覆盖奖励损失
    ├─ 目标: 高召回率 + 精确边界
    └─ 状态: 训练中
```

### 关键技术对比

| 特性 | V1 | V2 | V3 |
|------|----|----|-----|
| 标签来源 | Teacher模型预测 | TotalSegmentator文件 | TotalSegmentator文件 |
| 输入通道 | 3 (CT+HU特征+Teacher) | 2 (CT+标签) | 4 (CT+HU粗分割+标签+非排除) |
| 核心约束 | Teacher一致性 | HU硬约束 | 覆盖奖励+标签对齐 |
| 优化方向 | 平衡 | 精确率优先 | 召回率优先 |
| 腹部筛选 | 无 | 有 | 有 |
| 状态 | 失败 | 成功 | 训练中 |

---

## 文件清单

| 文件 | 版本 | 说明 |
|------|------|------|
| `scripts/train_muscle_transfer.py` | V1 | Teacher-Student架构（已废弃） |
| `scripts/train_muscle_transfer_v2.py` | V2 | TotalSegmentator标签 |
| `scripts/train_muscle_transfer_v3.py` | V3 | "先扩后缩"方法 |
| `outputs_muscle_transfer/` | V1 | V1输出目录 |
| `outputs_muscle_transfer_v2/` | V2 | V2输出目录 |
| `outputs_muscle_transfer_v3/` | V3 | V3输出目录 |

---

*报告生成时间: 2026-01-11*
