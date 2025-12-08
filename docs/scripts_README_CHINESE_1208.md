# 3D医学图像分割 - 训练与分析系统 (更新版)

本目录包含用于训练和分析CT扫描3D医学图像分割模型的Python脚本。

**最新更新 (2025-12-08)**：
- ✅ 添加早停机制 (Early Stopping)
- ✅ 添加准确率阈值自动停止
- ✅ 自动保存最佳模型
- ✅ 增强的训练监控和可视化

## 📋 脚本列表

### 核心训练脚本
- **`train_unet_enhanced.py`** - 增强版U-Net模型训练脚本（推荐）
  - 支持早停和准确率阈值
  - 实时监控（Weights & Biases）
  - 自动保存最佳模型
  - 详细的性能指标记录

- **`train_unet.py`** - 基础版U-Net模型训练脚本
  - 简单直接的训练流程
  - 适合快速实验

### 分析与评估脚本
- **`segmentation_detection_analysis.py`** - 模型评估与可视化脚本
  - 性能指标计算
  - 结果可视化
  - 错误分析

## 🎯 项目概述

本项目实现了基于2D U-Net的多器官分割模型，可从CT扫描中分割**117种不同的解剖结构**，包括：

### 主要器官系统
- **消化系统**：肝脏、胃、胰腺、脾脏、结肠、小肠、十二指肠、胆囊、食道
- **呼吸系统**：左/右肺上叶、中叶、下叶、气管
- **循环系统**：心脏、主动脉、上/下腔静脉、门静脉、肺静脉及各种血管分支
- **泌尿生殖系统**：左/右肾脏、肾囊肿、前列腺、膀胱、左/右肾上腺
- **神经系统**：脑、脊髓
- **内分泌系统**：甲状腺

### 骨骼系统
- **脊柱**：颈椎(C1-C7)、胸椎(T1-T12)、腰椎(L1-L5)、骶椎(S1)、骶骨
- **胸廓**：左/右肋骨(1-12)、胸骨、锁骨、肋软骨
- **四肢**：左/右肱骨、股骨、髋关节
- **其他**：头骨、肩胛骨

### 肌肉系统
- 臀大肌、臀中肌、臀小肌（左/右）
- 髂腰肌（左/右）
- 背部肌群（左/右）

## 🔧 系统要求

### 硬件要求
- **CPU**：多核处理器（推荐8核以上）
- **GPU**：支持CUDA的NVIDIA GPU（推荐）
  - 最小显存：8GB
  - 推荐显存：24GB以上
- **内存**：16GB以上（推荐32GB以上）
- **存储**：至少50GB可用空间

### 软件要求
- **操作系统**：Linux / macOS / Windows
- **Python**：3.10+
- **CUDA**：11.0+（如果使用GPU）

### Python依赖
所有依赖项列在 `requirements.txt` 中：
```
numpy>=1.21.0          # 数值计算
nibabel>=3.2.0         # NIfTI文件读写
scikit-image>=0.19.0   # 图像处理
torch>=2.0.0           # 深度学习框架
torchvision>=0.15.0    # 计算机视觉工具
matplotlib>=3.5.0      # 2D可视化
plotly>=5.0.0          # 3D交互式可视化
pandas>=1.3.0          # 数据分析
tqdm>=4.62.0           # 进度条显示
scipy>=1.7.0           # 科学计算
wandb>=0.15.0          # 实时训练监控
```

## 📦 快速开始

### 1. 安装依赖

```bash
cd /local/hzhang02/data/dataset
pip install -r requirements.txt
```

### 2. 准备数据

确保数据按以下结构组织：
```
/local/hzhang02/data/
├── s0000/
│   ├── ct.nii.gz
│   └── segmentations/
│       ├── liver.nii.gz
│       ├── heart.nii.gz
│       └── ... (117个分割文件)
├── s0001/
│   ├── ct.nii.gz
│   └── segmentations/
│       └── ...
└── ... (更多受试者)
```

### 3. 开始训练

**推荐方式（使用增强版脚本）**：
```bash
cd /local/hzhang02/data/dataset/scripts
python train_unet_enhanced.py
```

**基础方式**：
```bash
python train_unet.py
```

## ⚙️ 配置说明

### 基础配置（第374-380行）

```python
DATA_ROOT = '/local/hzhang02/data'           # 数据根目录
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs'  # 输出目录
TARGET_SHAPE = (256, 256)                    # 图像尺寸
BATCH_SIZE = 16                              # 批次大小
LEARNING_RATE = 1e-3                         # 初始学习率
EPOCHS = 20                                  # 最大训练轮数
```

### 早停配置（第382-385行）⭐ 新功能

```python
USE_EARLY_STOPPING = True                    # 启用早停
EARLY_STOP_PATIENCE = 5                      # 容忍轮数
EARLY_STOP_MIN_DELTA = 0.001                 # 最小改善阈值
```

**工作原理**：
- 监控验证集Dice系数
- 如果连续5个epoch提升小于0.1%，自动停止训练
- 防止过拟合和资源浪费

**推荐设置**：
- 快速实验：`PATIENCE = 3`
- 正常训练：`PATIENCE = 5`（默认）
- 精细调优：`PATIENCE = 7-10`

### 准确率阈值配置（第387-390行）⭐ 新功能

```python
USE_ACCURACY_THRESHOLD = True                # 启用阈值停止
ACCURACY_THRESHOLD = 0.93                    # 目标Dice系数
ACCURACY_THRESHOLD_PATIENCE = 2              # 稳定确认轮数
```

**工作原理**：
- 当验证Dice ≥ 0.93时开始计数
- 连续2个epoch都达到阈值后停止训练
- 确保性能稳定可靠

**阈值建议**：
- 快速验证：`0.85-0.90`
- 生产环境：`0.93-0.95`（推荐）
- 追求极致：`0.95-0.98`

### 学习率调度器配置（第392-401行）

```python
USE_SCHEDULER = True                         # 启用学习率调度
SCHEDULER_TYPE = 'cosine'                    # 'cosine' 或 'plateau'

# 余弦退火参数
COSINE_T_MAX = 20                            # 周期长度
COSINE_ETA_MIN = 1e-6                        # 最小学习率

# 自适应降低参数
PLATEAU_FACTOR = 0.5                         # 衰减因子
PLATEAU_PATIENCE = 3                         # 容忍轮数
PLATEAU_MIN_LR = 1e-6                        # 最小学习率
```

### Weights & Biases配置（第403-407行）

```python
USE_WANDB = True                             # 启用实时监控
WANDB_PROJECT = 'medical-segmentation-unet'  # 项目名称
WANDB_RUN_NAME = 'unet-2d-training-enhanced' # 运行名称
```

## 🚀 使用场景

### 场景1：快速实验（节省时间）

```python
# 修改配置
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 3

USE_ACCURACY_THRESHOLD = True
ACCURACY_THRESHOLD = 0.90
ACCURACY_THRESHOLD_PATIENCE = 1

BATCH_SIZE = 8  # 如果显存不足
```

**预期效果**：
- 训练时间：约6-8小时
- 预期停止：Epoch 8-10
- 适合：初步验证、快速迭代

### 场景2：标准训练（推荐）

```python
# 使用默认配置
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 5

USE_ACCURACY_THRESHOLD = True
ACCURACY_THRESHOLD = 0.93
ACCURACY_THRESHOLD_PATIENCE = 2

BATCH_SIZE = 16
```

**预期效果**：
- 训练时间：约7-9天
- 预期停止：Epoch 12-15
- 适合：生产环境、正式训练

### 场景3：追求最佳性能

```python
# 高质量配置
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 7

USE_ACCURACY_THRESHOLD = True
ACCURACY_THRESHOLD = 0.95
ACCURACY_THRESHOLD_PATIENCE = 3

SCHEDULER_TYPE = 'plateau'  # 更灵活的学习率调整
```

**预期效果**：
- 训练时间：约10-13天
- 预期停止：Epoch 15-18
- 适合：科研、高精度应用

## 📊 训练监控

### 控制台输出

```
============================================================
Epoch 12/20
============================================================
Training: 100%|██████████| 1250/1250 [2:15:30<00:00]
Validation: 100%|██████████| 312/312 [0:25:15<00:00]

Epoch 12 结果:
  训练损失: 0.0012 | 训练Dice: 0.9450
  验证损失: 0.0018 | 验证Dice: 0.9315 | 验证IoU: 0.9182
  过拟合差距: 0.0135 (正常)
  梯度范数: 0.0024
  Epoch耗时: 55432.5秒
  ✓ 验证Dice提升！新的最佳: 0.9315
  ✓ 已达到准确率阈值 0.9300！(2/2轮)
  学习率已更新: 0.00034567

  表现最好的5个结构:
    liver: 0.9856
    spleen: 0.9782
    kidney_left: 0.9745
    kidney_right: 0.9723
    heart: 0.9698

  表现最差的5个结构:
    rib_left_12: 0.7234
    vertebrae_C1: 0.7456
    thyroid_gland: 0.7589
    prostate: 0.7623
    adrenal_gland_left: 0.7701

============================================================
准确率阈值停止触发！
  验证Dice已达到 0.9315 >= 0.9300
  并稳定保持了 2 个epoch
============================================================
```

### Weights & Biases 实时监控

训练开始后，访问 wandb.ai 查看：
- 实时训练/验证曲线
- 学习率变化
- 梯度范数监控
- 每个类别的性能
- 样本预测可视化
- 系统资源使用情况

### 输出文件

训练过程会自动生成以下文件：

```
outputs/
├── checkpoint_enhanced_epoch{N}.pth      # 每轮检查点
├── best_model.pth                        # 最佳模型 ⭐
├── training_history_enhanced.json        # 训练历史数据
├── training_history_enhanced.png         # 训练曲线图
├── test_inference_enhanced.png           # 测试推理结果
├── sample_visualization_enhanced.png     # 样本可视化
├── label_map.json                        # 标签映射
├── ct_slices.png                         # CT切片展示
└── ct_mesh.html                          # 3D网格可视化
```

## 🔍 模型评估

### 加载最佳模型

```python
import torch
from scripts.train_unet_enhanced import UNet2D

# 加载模型
checkpoint = torch.load('outputs/best_model.pth')
model = UNet2D(in_ch=1, out_ch=117, features=[32, 64, 128, 256])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"最佳Dice系数: {checkpoint['val_dice']:.4f}")
print(f"最佳IoU: {checkpoint['val_iou']:.4f}")
print(f"训练轮数: {checkpoint['epoch']}")
```

### 性能指标

根据实际训练数据：

| 指标 | 初始值 | 最终值 | 提升 |
|------|--------|--------|------|
| 训练Dice | 0.8200 | 0.9764 | +19.1% |
| 验证Dice | 0.8358 | 0.9317 | +11.5% |
| 验证IoU | 0.8356 | 0.9184 | +9.9% |
| 训练损失 | 0.0604 | 0.0003 | -99.5% |

## ⏱️ 性能优化

### 时间节省

| 配置 | 预期轮数 | 训练时长 | 节省 |
|------|---------|---------|------|
| 无早停 | 20 | ~308小时 | 0% |
| 阈值0.90 | 10-12 | ~170小时 | 45% |
| 阈值0.93 | 12-15 | ~215小时 | 30% |
| 阈值0.95 | 15-18 | ~260小时 | 15% |

### 加速训练技巧

1. **增加批次大小**（如果显存允许）
   ```python
   BATCH_SIZE = 32  # 从16增加到32
   ```

2. **使用混合精度训练**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

3. **增加数据加载线程**
   ```python
   num_workers=8  # 从4增加到8
   ```

4. **使用更快的学习率调度器**
   ```python
   SCHEDULER_TYPE = 'plateau'  # 比cosine更灵活
   ```

## 📚 相关文档

- `docs/training_report_detailed.md` - 详细训练报告
- `docs/training_issues_and_improvements.md` - 问题分析与改进建议
- `docs/early_stopping_guide.md` - 早停机制使用指南
- `docs/MODEL_GUIDE_CHINESE_1114.md` - 模型架构详解

## 🔧 故障排查

### 问题1：显存不足 (Out of Memory)

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：
```python
BATCH_SIZE = 8  # 减小批次大小
TARGET_SHAPE = (128, 128)  # 减小图像尺寸
```

### 问题2：训练过早停止

**症状**：只训练了2-3个epoch就停止

**解决方案**：
```python
USE_ACCURACY_THRESHOLD = False  # 暂时禁用阈值
# 或
ACCURACY_THRESHOLD = 0.95  # 提高阈值
ACCURACY_THRESHOLD_PATIENCE = 3  # 增加确认轮数
```

### 问题3：训练速度慢

**症状**：每个epoch需要很长时间

**可能原因与解决**：
- 数据加载慢：增加 `num_workers`
- CPU瓶颈：减少数据增强操作
- 磁盘I/O慢：将数据复制到SSD
- GPU利用率低：增加 `BATCH_SIZE`

### 问题4：Weights & Biases连接失败

**解决方案**：
```python
USE_WANDB = False  # 禁用wandb
# 或手动登录
import wandb
wandb.login(key='your-api-key')
```

### 问题5：找不到数据

**症状**：`找到 0 个受试者`

**解决方案**：
```python
# 检查数据路径
DATA_ROOT = '/local/hzhang02/data'  # 确保路径正确
# 检查文件夹命名（必须以's'开头，如s0000, s0001）
```

## 💡 最佳实践

### 1. 训练前检查清单

- [ ] 数据路径正确
- [ ] 显存足够（至少8GB）
- [ ] 硬盘空间充足（至少50GB）
- [ ] 早停和阈值配置合理
- [ ] Weights & Biases已配置（可选）

### 2. 训练中监控

- [ ] 定期查看wandb曲线
- [ ] 监控GPU利用率
- [ ] 检查磁盘空间
- [ ] 观察验证Dice趋势

### 3. 训练后分析

- [ ] 查看训练历史图表
- [ ] 分析最佳/最差类别
- [ ] 检查过拟合情况
- [ ] 保存最佳模型

## 📞 技术支持

如遇到问题，请检查：
1. 相关文档（`docs/`目录）
2. 训练日志输出
3. Weights & Biases报告
4. GPU/内存使用情况

## 📄 许可证

本项目仅供学术研究使用。

---

**最后更新**：2025-12-08
**版本**：v2.0（增强版）
**维护者**：hzhang02
