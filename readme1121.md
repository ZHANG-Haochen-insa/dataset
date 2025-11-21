# 训练脚本更新日志 - 2025-11-21

## 📝 修改摘要

今天创建了一个**带动态学习率调度器的增强版训练脚本**，可以在wandb上实时监控学习率变化。

---

## 🆕 新增文件

### `train_unet_enhanced.py`
- 位置：`/local/hzhang02/data/dataset/scripts/train_unet_enhanced.py`
- 功能：带动态学习率和完整监控的U-Net训练脚本

---

## ⚡ 主要改进

### 1. **动态学习率调度器** ⭐⭐⭐

#### 支持两种学习率调度策略：

**A. 余弦退火 (CosineAnnealingLR) - 当前使用 ✅**
```python
SCHEDULER_TYPE = 'cosine'
COSINE_T_MAX = 20  # 总epoch数
COSINE_ETA_MIN = 1e-6  # 最小学习率
```

**学习率变化曲线：**
```
0.001 (开始)
      ╲╲╲╲╲
           ╲╲╲╲
                ╲╲╲___
                       0.000001 (结束)
epoch 1 → → → → → → → → → 20
```

**特点：**
- ✅ 平滑的余弦曲线下降
- ✅ 开始快速学习，后期精细调整
- ✅ 适合固定训练轮数的场景
- ✅ 训练稳定，效果好

---

**B. 自适应降低 (ReduceLROnPlateau) - 备选**
```python
SCHEDULER_TYPE = 'plateau'
PLATEAU_FACTOR = 0.5  # 每次降低到原来的50%
PLATEAU_PATIENCE = 3  # 容忍3个epoch性能不提升
PLATEAU_MIN_LR = 1e-6  # 最小学习率
```

**学习率变化逻辑：**
```
如果连续3个epoch验证Dice不提升：
  学习率 = 学习率 × 0.5
```

**特点：**
- ✅ 自动根据训练效果调整
- ✅ 当性能停滞时降低学习率
- ✅ 适合不确定训练轮数的场景

---

### 2. **性能优化** 🚀

| 参数 | 原版 | 增强版 | 提升 |
|------|------|--------|------|
| **Batch Size** | 8 | 16 | 2倍 |
| **Num Workers** | 0 | 4 | 并行加载 |
| **Pin Memory** | False | True | GPU传输加速 |
| **训练轮数** | 5 | 20 | 更充分训练 |
| **速度** | 70秒/batch | 预计7-14秒/batch | **5-10倍** ⚡ |

**预计训练时间：**
- 1个epoch: 4-6小时（vs 原来40小时）
- 20个epoch: 80-120小时（3.5-5天）

---

### 3. **wandb监控指标增强** 📊

#### 新增监控指标：

**训练质量指标：**
1. ✅ `train/dice` - 训练集Dice分数
2. ✅ `train/loss` - 训练集损失
3. ✅ `val/dice` - 验证集Dice分数
4. ✅ `val/loss` - 验证集损失
5. ✅ `val/iou` - 验证集IoU分数

**对比分析指标：**
6. ✅ `comparison/loss_gap` - 训练loss - 验证loss
7. ✅ `comparison/dice_gap` - 训练dice - 验证dice
8. ✅ `comparison/overfit_score` - 过拟合分数

**学习率监控：** ⭐⭐⭐
9. ✅ `learning_rate` - 每个epoch的学习率（实时变化）

**训练状态指标：**
10. ✅ `gradient_norm` - 梯度范数（判断训练稳定性）
11. ✅ `epoch_time` - 每个epoch耗时
12. ✅ `samples_per_second` - 训练速度

**类别性能指标：**
13. ✅ `top_classes/rank_1_xxx` - 表现最好的5个器官
14. ✅ `bottom_classes/rank_1_xxx` - 表现最差的5个器官

**批次级监控：**
15. ✅ `batch/loss` - 每10个batch记录一次
16. ✅ `batch/dice` - 每10个batch记录一次
17. ✅ `batch/gradient_norm` - 每10个batch记录一次

---

### 4. **wandb Config记录** 📋

所有训练配置都记录在wandb的Config标签页：

```yaml
# 基础配置
architecture: "U-Net 2D"
epochs: 20
batch_size: 16
learning_rate: 0.001

# 学习率调度器配置 ⭐
use_scheduler: true
scheduler_type: "cosine"
cosine_t_max: 20
cosine_eta_min: 0.000001

# 数据集信息
num_classes: 117
train_subjects: 64
val_subjects: 17
train_slices: 16661
val_slices: 4461
train_ratio: 0.8
val_ratio: 0.2

# 性能优化
num_workers: 4
pin_memory: true

# 模型信息
total_params: 7769237
trainable_params: 7769237
```

---

## 📊 如何在wandb上查看学习率

### 方法1：在Charts标签页查看

1. 打开你的wandb项目页面
2. 点击左侧的 **Charts** 标签
3. 在指标列表中找到 `learning_rate`
4. 点击即可看到学习率随epoch的变化曲线

**你会看到的曲线（余弦退火）：**
```
Learning Rate
0.001 ┐
      │╲╲╲
      │   ╲╲╲
      │      ╲╲╲
      │         ╲╲╲___
0.000001 ┘────────────────
        1  5  10  15  20 (epoch)
```

### 方法2：在Workspace中创建自定义图表

1. 点击 **Workspace** 标签
2. 点击 **+ Add visualization**
3. 选择 **Line plot**
4. X轴选择：`epoch`
5. Y轴选择：`learning_rate`
6. 可以同时添加 `val_dice` 来对比学习率和性能的关系

### 方法3：对比学习率和性能

创建双Y轴图表：
- 左Y轴：`learning_rate`（对数刻度）
- 右Y轴：`val_dice`（线性刻度）
- X轴：`epoch`

这样可以直观看到学习率降低时，模型性能的变化。

---

## 🎯 如何判断训练好不好？

### ✅ 理想的训练状态

1. **损失曲线：**
   ```
   train_loss: ╲╲╲╲╲____  (持续下降后平稳)
   val_loss:   ╲╲╲╲╲____  (跟随下降)
   两条线很接近 ✅
   ```

2. **Dice曲线：**
   ```
   train_dice: ____╱╱╱╱╱  (持续上升后平稳)
   val_dice:   ____╱╱╱╱╱  (跟随上升)
   两条线很接近 ✅
   ```

3. **学习率变化：**
   ```
   learning_rate: 0.001 → 0.000001 (平滑下降) ✅
   ```

4. **性能指标：**
   - val_dice > 0.7 👍 不错
   - val_dice > 0.8 🎉 很好
   - val_dice > 0.85 🏆 优秀
   - overfit_score < 0.1 ✅ 泛化好

---

### ⚠️ 需要注意的问题

1. **过拟合（Overfitting）：**
   ```
   train_dice: ____╱╱╱╱╱  (高)
   val_dice:   ____╱╱___   (不再上升)
   两条线分开越来越远 ⚠️

   解决方法：
   - 增加数据增强
   - 减小模型复杂度
   - 使用Dropout
   ```

2. **学习率过大：**
   ```
   loss: ╱╲╱╲╱╲╱╲  (震荡)
   gradient_norm: 波动很大

   解决方法：
   - 降低初始学习率
   - 使用学习率调度器 ✅（已使用）
   ```

3. **学习率过小：**
   ```
   loss: ━━━━━━━  (不下降)
   val_dice: ━━━━  (不上升)

   解决方法：
   - 提高初始学习率
   - 检查数据预处理
   ```

4. **梯度问题：**
   ```
   gradient_norm > 100  ⚠️ 梯度爆炸
   gradient_norm < 0.01 ⚠️ 梯度消失

   正常范围：0.1 - 10 ✅
   ```

---

## 🚀 如何开始训练

### 1. 启动训练（后台运行）

```bash
cd /local/hzhang02/data/dataset/scripts
nohup python -u train_unet_enhanced.py > training_enhanced.log 2>&1 &
```

### 2. 查看实时日志

```bash
tail -f training_enhanced.log
```

### 3. 查看wandb实时监控

打开浏览器访问：
```
https://wandb.ai/haochen-zhang-insa-lyon/medical-segmentation-unet
```

---

## 📁 输出文件

训练完成后会生成以下文件（位于 `/local/hzhang02/data/dataset/outputs/`）：

### 模型检查点：
```
checkpoint_enhanced_epoch1.pth
checkpoint_enhanced_epoch2.pth
...
checkpoint_enhanced_epoch20.pth
```

每个检查点包含：
- 模型权重
- 优化器状态
- 训练/验证指标
- 当前epoch信息

### 训练历史：
```
training_history_enhanced.json    # 完整的训练历史数据
training_history_enhanced.png     # 6张训练曲线图
```

6张曲线图包括：
1. 训练Loss vs 验证Loss
2. 训练Dice vs 验证Dice
3. 验证IoU
4. 过拟合分析（Dice差距）
5. 梯度范数
6. **学习率变化曲线** ⭐

### 可视化结果：
```
sample_visualization_enhanced.png     # 训练样本可视化
test_inference_enhanced.png          # 测试推理结果
```

### 配置文件：
```
label_map.json  # 117个解剖结构的标签映射
```

---

## 🔧 自定义配置

如果想修改学习率调度器，编辑 `train_unet_enhanced.py` 的配置部分：

### 切换到Plateau调度器：

```python
# 学习率调度器配置
USE_SCHEDULER = True
SCHEDULER_TYPE = 'plateau'  # 改成 'plateau'

# Plateau参数（取消注释并调整）
PLATEAU_FACTOR = 0.5      # 衰减因子
PLATEAU_PATIENCE = 3      # 容忍轮数
PLATEAU_MIN_LR = 1e-6     # 最小学习率
```

### 调整余弦退火参数：

```python
SCHEDULER_TYPE = 'cosine'
COSINE_T_MAX = 20         # 修改为实际的总epoch数
COSINE_ETA_MIN = 1e-6     # 最小学习率（可以改成1e-7更小）
```

### 关闭学习率调度器：

```python
USE_SCHEDULER = False  # 使用固定学习率
```

---

## 📊 Wandb页面解读

### Config标签页
- 查看所有训练配置
- 包括学习率调度器类型和参数

### Charts标签页
主要关注这些曲线：
1. **learning_rate** - 学习率变化 ⭐
2. **val/dice** - 验证集性能（最重要）
3. **comparison/overfit_score** - 过拟合程度
4. **gradient_norm** - 训练稳定性

### Workspace标签页
- 可以创建自定义图表
- 建议创建：学习率 vs 性能对比图

### System标签页
- GPU使用率
- 内存占用
- 系统资源监控

---

## 📈 预期训练效果

根据医学图像分割的经验：

### 前5个epoch (学习率高: 0.001 → 0.0005)
- Loss快速下降
- Dice快速上升 (0 → 0.5-0.7)
- 学习速度快

### 中间10个epoch (学习率中: 0.0005 → 0.0001)
- Loss缓慢下降
- Dice缓慢上升 (0.7 → 0.8)
- 稳定提升

### 最后5个epoch (学习率低: 0.0001 → 0.000001)
- Loss微调
- Dice精细提升 (0.8 → 0.82)
- 细节优化

### 最终目标：
- **val_dice > 0.75** - 合格 ✅
- **val_dice > 0.80** - 良好 👍
- **val_dice > 0.85** - 优秀 🎉

---

## 🆚 脚本对比

| 特性 | train_unet.py | train_unet_enhanced.py |
|------|---------------|------------------------|
| 学习率 | 固定 | **动态调整** ⭐ |
| Batch size | 8 | 16 |
| 数据加载 | 同步 | 并行 |
| 训练轮数 | 5 | 20 |
| 监控指标 | 5个 | 17个 |
| 学习率监控 | ✅ | ✅ + 动态变化曲线 |
| 梯度监控 | ❌ | ✅ |
| IoU监控 | ❌ | ✅ |
| 类别性能 | ❌ | ✅ Top/Bottom 5 |
| 过拟合分析 | ❌ | ✅ |
| 速度 | 慢（70秒/batch） | 快（7-14秒/batch） |
| 推荐使用 | 测试 | **生产训练** ⭐ |

---

## 💡 训练建议

### 第一次训练：
1. 使用增强版脚本
2. 保持默认配置（余弦退火）
3. 训练20个epoch
4. 观察wandb上的学习率曲线

### 如果性能不理想：
1. 检查过拟合分数（应该 < 0.1）
2. 查看学习率是否下降过快
3. 尝试调整 `COSINE_ETA_MIN`（最小学习率）
4. 或切换到Plateau调度器

### 如果训练很慢：
1. 检查 `num_workers` 是否生效
2. 检查GPU使用率（应该 > 80%）
3. 可以进一步增加 `BATCH_SIZE` 到24或32

---

## 🔍 故障排除

### 问题1：学习率在wandb上显示为固定值
**原因：** 未启用调度器
**解决：** 确认 `USE_SCHEDULER = True`

### 问题2：学习率下降太快
**原因：** `COSINE_ETA_MIN` 太小或 `COSINE_T_MAX` 太小
**解决：**
```python
COSINE_ETA_MIN = 1e-5  # 改大一点
COSINE_T_MAX = 30      # 改大一点
```

### 问题3：性能不提升但学习率已经很小
**原因：** 可能陷入局部最优
**解决：**
- 重新训练，使用更大的初始学习率
- 或使用Plateau调度器

### 问题4：训练速度没有提升
**原因：** 数据加载瓶颈
**解决：**
```python
# 检查num_workers是否生效
# 可以尝试增加到6或8
train_loader = DataLoader(..., num_workers=6)
```

---

## 📞 相关文件

- **训练脚本：** `/local/hzhang02/data/dataset/scripts/train_unet_enhanced.py`
- **评估脚本：** `/local/hzhang02/data/dataset/scripts/segmentation_detection_analysis.py`
- **输出目录：** `/local/hzhang02/data/dataset/outputs/`
- **训练日志：** `/local/hzhang02/data/dataset/scripts/training_enhanced.log`

---

## 🎓 总结

这次更新的核心是：

1. ⭐ **动态学习率调度器** - 在wandb上可视化学习率变化
2. 🚀 **性能优化** - 训练速度提升5-10倍
3. 📊 **监控增强** - 17个详细指标，全方位监控训练
4. 🎯 **智能调整** - 自动根据训练情况调整学习率

**核心优势：**
- 可以在wandb上实时看到学习率如何变化
- 可以判断学习率调整是否合理
- 可以对比学习率和模型性能的关系
- 训练过程更加透明和可控

---

**创建时间：** 2025-11-21
**作者：** Claude Code Assistant
**版本：** v2.0 Enhanced with Dynamic Learning Rate
