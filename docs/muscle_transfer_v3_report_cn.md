# 肌肉迁移学习 V3 改动报告

## 概述

本报告记录了从V2版本升级到V3版本"先扩后缩"方法的完整过程，包括问题分析、设计思路和具体实现。

## 背景与问题

### V2版本的局限性

V2迁移学习版本在验证集上取得了不错的指标（HU符合率99.5%，Dice 0.80），但在实际分析中发现了核心问题：

**模型过于保守**：模型预测的肌肉区域覆盖面积不足，即使是已标注的肌肉样本区域也没有被完全覆盖。这与我们的最终目标——发现和分割所有肌肉组织——相悖。

### 用户需求

用户明确指出希望模型能够：
1. **覆盖更多的肌肉区域**，而非局限于训练样本中已标注的部分
2. 通过HU值范围先确定所有可能的肌肉候选区域
3. 再通过与已知标签的对齐来精确调整边界

## 设计思路："先扩后缩"方法

### 核心理念

```
先扩：覆盖所有HU值在肌肉范围内的像素（高召回率）
后缩：通过与已知标签对齐来精确边界（高精确率）
```

### 方法对比

| 特性 | V2版本 | V3版本 |
|------|--------|--------|
| 输入通道 | 2（CT + 已知标签） | 4（CT + HU粗分割 + 已知标签 + 非排除区域） |
| 损失设计 | BCE + HU合规 + Dice | 标签对齐 + 覆盖奖励 + 排除惩罚 |
| 目标 | 精确匹配已知标签 | 最大化覆盖同时保持边界精度 |
| 优化方向 | 精确率优先 | 召回率优先，边界精确 |

## 具体改动

### 1. 输入设计（4通道）

```python
input_tensor = np.stack([
    ct_resized,           # 通道0: CT图像（归一化后）
    hu_coarse,            # 通道1: HU粗分割（所有HU在-29~150的像素）
    muscle_resized,       # 通道2: 已知肌肉标签
    1 - exclusion         # 通道3: 非排除区域掩码
], axis=0)
```

**设计原因**：
- `hu_coarse`提供了肌肉的物理候选区域，告诉模型"这些像素在物理上可能是肌肉"
- `muscle_resized`提供了已知的正确标签，用于边界学习
- `1 - exclusion`明确标出了不可能是肌肉的区域（空气、骨骼）

### 2. 损失函数设计

```python
class ExpandThenRefireLoss(nn.Module):
    """
    先扩后缩损失函数

    目标：
    1. 在已标注区域精确匹配（高权重）
    2. 在未标注但HU有效区域鼓励覆盖（覆盖奖励）
    3. 严格排除空气和骨骼区域
    """
```

#### 损失组成

| 损失项 | 权重 | 作用 |
|--------|------|------|
| label_alignment | 3.0 | 确保与已知肌肉标签精确对齐 |
| coverage_reward | 1.0 | 鼓励覆盖HU有效但未标注的区域 |
| exclusion | 5.0 | 严格惩罚预测为空气/骨骼的错误 |

#### 关键代码逻辑

```python
# 1. 已标注区域对齐损失 - 高权重确保精度
label_region_loss = (bce_all * has_label).sum() / (has_label.sum() + 1e-6)
losses['label_alignment'] = label_region_loss * 3.0

# 2. 覆盖奖励 - 鼓励覆盖未标注但HU有效的区域
unlabeled_hu_valid = hu_coarse * (1 - has_label) * body_mask * (1 - exclusion_mask)
coverage_loss = ((1 - pred_prob) * unlabeled_hu_valid).mean()
losses['coverage_reward'] = coverage_loss * 1.0

# 3. 排除区域损失 - 绝对禁止预测空气/骨骼
exclusion_loss = (pred_prob * exclusion_mask).mean()
losses['exclusion'] = exclusion_loss * 5.0
```

### 3. 训练配置

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
LABEL_ALIGNMENT_WEIGHT = 3.0
COVERAGE_REWARD_WEIGHT = 1.0
EXCLUSION_WEIGHT = 5.0
```

### 4. 腹部筛选

保留了V2中的腹部筛选逻辑，只处理腹部区域的切片（排除大腿部分）：

```python
# 通过解剖标记判断是否为腹部区域
abdominal_organs = [1, 2, 3, 5, 6]  # 脾脏、肾脏、肝脏等
thigh_bones = [74, 76]  # 左右股骨

is_abdominal = any(organ in labels for organ in abdominal_organs)
has_thigh = any(bone in labels for bone in thigh_bones)

# 保留腹部切片，排除大腿切片
keep_slice = is_abdominal and not has_thigh
```

## 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/train_muscle_transfer_v3.py` | 新建 | V3"先扩后缩"方法的完整实现 |
| `.gitignore` | 修改 | 添加`outputs_muscle_transfer_v3/`的规则 |
| `outputs_muscle_transfer_v3/` | 新建 | V3训练输出目录 |

## 训练状态

- **启动时间**: 2026-01-11 01:08
- **进程PID**: 2333264
- **训练数据**: 6,352个腹部切片（30个受试者）
- **验证数据**: 2,119个腹部切片（8个受试者）
- **WandB链接**: https://wandb.ai/haochen-zhang-insa-lyon/muscle-transfer-learning/runs/ok1dch31

## 预期效果

1. **更高的召回率**：模型应该能覆盖更多的肌肉区域
2. **保持边界精度**：通过label_alignment损失，已知区域的边界应该保持准确
3. **避免假阳性**：exclusion损失确保空气和骨骼不会被误判

## 后续改进方向

1. 如果覆盖率仍不够，可以增加`COVERAGE_REWARD_WEIGHT`
2. 如果边界不够精确，可以增加`LABEL_ALIGNMENT_WEIGHT`
3. 考虑添加连通性约束，避免孤立的小区域

---

*报告生成时间: 2026-01-11*
