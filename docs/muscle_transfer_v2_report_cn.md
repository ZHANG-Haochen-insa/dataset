# 肌肉迁移学习 V2 改动报告

## 概述

本报告记录了从V1版本升级到V2版本的完整过程，包括问题分析、设计思路和具体实现。

## 背景与问题

### V1版本的问题

V1版本采用了Teacher-Student架构，使用预训练的肌肉模型作为Teacher来生成伪标签。但在实际运行中遇到了严重问题：

**模型结构不匹配**：原始训练的UNet2D模型与V1代码中定义的Teacher模型结构不一致，导致无法正确加载权重。

```
错误信息示例：
Unexpected key(s) in state_dict: "encoder.0.net.3.weight"...
```

原因是原始模型的`DoubleConv`模块包含Dropout层，而V1代码中没有。

### 设计缺陷

1. **依赖模型预测**：使用模型预测作为伪标签，可能传播错误
2. **复杂的Teacher-Student架构**：增加了调试难度
3. **缺乏硬约束**：对空气和骨骼区域没有足够强的排除机制

## V2解决方案

### 核心改进

```
直接使用TotalSegmentator的真实分割标签
↓
不再依赖预训练模型预测
↓
增加HU值硬约束
↓
添加腹部区域筛选
```

### 方法对比

| 特性 | V1版本 | V2版本 |
|------|--------|--------|
| 标签来源 | 预训练模型预测（伪标签） | TotalSegmentator真实分割 |
| 架构 | Teacher-Student | 单一Student模型 |
| HU约束 | 软约束 | 硬约束（空气/骨骼必须排除） |
| 区域筛选 | 无 | 腹部筛选（排除大腿） |

## 具体改动

### 1. 移除Teacher模型依赖

V1需要加载预训练模型：
```python
# V1 - 需要加载Teacher模型
teacher = DoubleConvTeacher(...)  # 结构必须完全匹配
teacher.load_state_dict(checkpoint['model_state_dict'])
```

V2直接使用标签文件：
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

### 2. 增强HU约束

```python
# 硬排除区域定义
AIR_HU_MAX = -200      # 空气/背景，必须排除
BONE_HU_MIN = 300      # 骨骼，必须排除

# 损失权重配置
LABEL_WEIGHT = 2.0            # 已标注区域一致性
HU_CONSTRAINT_WEIGHT = 3.0    # HU约束权重
EXCLUSION_WEIGHT = 5.0        # 排除区域约束（最强！）
BOUNDARY_WEIGHT = 0.3         # 边界平滑
```

### 3. 新的损失函数设计

```python
class MuscleTransferLossV2(nn.Module):
    """
    四项损失：
    1. label - 已标注区域一致性
    2. exclusion - 空气/骨骼区域必须预测为0
    3. hu_constraint - HU值范围约束
    4. boundary - 边界平滑
    """

    def forward(self, pred_logits, label_mask, hu_slice, body_mask, exclusion_mask):
        # 1. 标签一致性损失
        label_loss = (bce_loss * weight * body_mask).sum() / (body_mask.sum() + 1e-6)

        # 2. 排除区域损失（最重要！）
        exclusion_loss = (pred_prob * exclusion_mask).mean()

        # 3. HU约束损失
        hu_violation = pred_prob * hu_invalid * body_mask * (1 - exclusion_mask)
        hu_loss = hu_violation.mean() * 10

        # 4. 边界平滑损失
        tv_loss = tv_h + tv_w
```

### 4. 腹部区域筛选

用户反馈希望模型专注于腹部区域，而非大腿部分。V2添加了解剖学筛选：

```python
# 腹部器官标记（有这些器官的切片是腹部）
ABDOMINAL_ORGANS = [
    'liver.nii.gz',
    'spleen.nii.gz',
    'kidney_left.nii.gz',
    'kidney_right.nii.gz',
    'pancreas.nii.gz',
    'stomach.nii.gz',
    'colon.nii.gz',
    'small_bowel.nii.gz',
]

# 排除标记（有这些骨骼的切片是大腿）
EXCLUDE_MARKERS = [
    'femur_left.nii.gz',
    'femur_right.nii.gz',
]

# 筛选逻辑
has_abdominal = any(organ present in slice)
has_thigh = any(femur present in slice)
keep_slice = has_abdominal and not has_thigh
```

**筛选效果**：
- 训练集：6352/7695 切片保留（82.5%）
- 验证集：2119/2540 切片保留（83.4%）

### 5. 输入设计（2通道）

```python
input_tensor = np.stack([
    ct_resized,           # 通道0: CT图像（归一化后）
    muscle_resized,       # 通道1: 已知肌肉标签
], axis=0)
```

### 6. WandB集成

添加了完整的WandB日志记录：
- 训练/验证损失曲线
- 各损失分量追踪
- 验证指标（Dice, HU合规率）
- 定期可视化样本

## 训练结果

V2训练在验证集上取得了良好的指标：

| 指标 | 数值 |
|------|------|
| HU符合率 | 99.5% |
| Dice分数 | 0.80 |
| 标签损失 | 稳定下降 |
| 排除损失 | 接近0 |

## 发现的问题（导致V3）

尽管V2的指标不错，但实际分析发现：

**模型过于保守**：预测的肌肉区域覆盖面积不足，即使是已标注的肌肉区域也没有被完全覆盖。

这个问题促使了V3"先扩后缩"方法的设计。

## 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/train_muscle_transfer_v2.py` | 新建 | V2完整实现 |
| `.gitignore` | 修改 | 添加`outputs_muscle_transfer_v2/`规则 |
| `outputs_muscle_transfer_v2/` | 新建 | V2训练输出目录 |

## 技术债务清理

V2还修复了V1中的多个技术问题：

1. **CUDA多进程错误**：设置`num_workers=0`解决"Cannot re-initialize CUDA in forked subprocess"
2. **JSON序列化错误**：将numpy类型转换为Python原生类型
3. **模型结构匹配**：不再需要匹配预训练模型结构

---

*报告生成时间: 2026-01-11*
