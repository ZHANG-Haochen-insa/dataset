# 医学图像分割模型训练指南（云服务器实时监控版）

**版本**: 1121 | **更新日期**: 2025年11月21日

---

## 目录

1. [项目简介](#项目简介)
2. [云服务器实时监控功能](#云服务器实时监控功能)
3. [快速开始](#快速开始)
4. [详细使用指南](#详细使用指南)
5. [实时监控配置](#实时监控配置)
6. [监控面板使用](#监控面板使用)
7. [常见问题](#常见问题)
8. [最佳实践](#最佳实践)

---

## 项目简介

本项目实现了基于2D U-Net的医学图像分割模型，用于从CT扫描中分割117个解剖结构。

### 主要特性

- **多器官分割**: 支持117个解剖结构的自动分割
- **2D U-Net架构**: 高效的编码器-解码器结构
- **实时监控**: 集成Weights & Biases云端监控
- **云服务器优化**: 适合长时间训练任务的实时追踪

### 数据集信息

- **格式**: NIfTI (.nii.gz)
- **输入**: CT扫描图像
- **输出**: 117通道的分割掩码
- **分割结构**: 包括主要器官、骨骼、血管系统、肌肉群等

---

## 云服务器实时监控功能

### 为什么需要实时监控？

在云服务器上进行长时间的深度学习训练时，实时监控至关重要：

✅ **随时随地查看进度**: 无需登录服务器即可查看训练状态
✅ **及时发现问题**: 快速识别训练异常（如损失爆炸、梯度消失）
✅ **优化训练策略**: 根据实时指标调整超参数
✅ **记录完整历史**: 自动保存所有训练指标和可视化结果
✅ **多设备访问**: 手机、平板、电脑都可以查看

### 监控内容

我们的实时监控系统会记录以下内容：

#### 📊 训练指标
- 每个batch的训练损失
- 每个epoch的平均训练损失
- 验证集Dice系数
- 学习率变化

#### 🖼️ 可视化内容
- 训练样本预测对比图
- CT图像与分割掩码叠加
- 模型预测结果实时更新

#### 🔧 系统信息
- GPU使用情况
- 训练速度（samples/sec）
- 模型梯度和参数统计

#### 🏆 最佳性能跟踪
- 最高验证Dice系数
- 最佳模型对应的epoch
- 历史最优记录

---

## 快速开始

### 第一步：环境准备

```bash
# 1. 进入项目目录
cd /home/zhanghaochen/Desktop/cours/TDSI/pjt_1/dataset

# 2. 安装必要的Python包（如果还没安装）
pip install wandb torch torchvision numpy nibabel scikit-image matplotlib tqdm
```

### 第二步：配置Weights & Biases

```bash
# 1. 注册wandb账号（如果还没有）
# 访问 https://wandb.ai/ 注册一个免费账号

# 2. 登录wandb
wandb login
# 这会提示你输入API Key，从 https://wandb.ai/authorize 获取
```

**重要提示**:
- API Key只需要配置一次，之后会自动保存
- 在云服务器上配置后，所有训练任务都会自动上传数据
- 即使SSH断开连接，数据仍会继续上传

### 第三步：启动训练并监控

```bash
# 1. 启动Jupyter Lab
cd /home/zhanghaochen/Desktop/cours/TDSI/pjt_1/dataset/scripts
jupyter lab train_unet.ipynb

# 2. 在notebook中运行所有单元格
# 训练开始后会自动输出监控链接

# 3. 打开监控链接
# 控制台会显示类似这样的链接：
# 🚀 查看实时训练进度: https://wandb.ai/your-username/medical-segmentation-unet/runs/xxx
```

### 第四步：查看实时监控

在浏览器中打开上述链接，你就能看到：
- 📈 实时更新的损失曲线
- 🎯 验证指标变化趋势
- 🖼️ 模型预测结果样本
- 💻 系统资源使用情况

---

## 详细使用指南

### 1. 修改训练配置

在 `train_unet.ipynb` 的配置单元格中，你可以调整以下参数：

```python
# ==================== 基础配置 ====================
DATA_ROOT = '/local/hzhang02/data'          # 数据集根目录
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs'  # 输出目录
TARGET_SHAPE = (256, 256)                   # 图像尺寸
BATCH_SIZE = 8                              # 批次大小（根据GPU显存调整）
LEARNING_RATE = 1e-3                        # 学习率
EPOCHS = 50                                 # 训练轮数（长期训练建议50-100）

# ==================== 实时监控配置 ====================
USE_WANDB = True                            # 是否启用实时监控
WANDB_PROJECT = 'medical-segmentation-unet' # 项目名称
WANDB_RUN_NAME = 'unet-2d-training'         # 运行名称
WANDB_NOTES = '2D U-Net training for 117 anatomical structures'

# 日志记录频率
LOG_EVERY_N_BATCHES = 10                    # 每10个batch记录一次
LOG_IMAGES_EVERY_N_EPOCHS = 1               # 每1个epoch记录图像
```

### 2. 长时间训练建议配置

对于在云服务器上的长时间训练（几小时到几天），推荐以下配置：

```python
EPOCHS = 100                    # 增加训练轮数
BATCH_SIZE = 16                 # 如果GPU内存足够，增大batch size
LOG_EVERY_N_BATCHES = 20        # 减少日志频率，节省带宽
LOG_IMAGES_EVERY_N_EPOCHS = 5   # 每5个epoch记录一次图像
```

### 3. 后台运行训练

为了防止SSH断开导致训练中断，建议使用以下方法之一：

#### 方法A: 使用tmux（推荐）

```bash
# 1. 安装tmux（如果还没有）
sudo apt-get install tmux

# 2. 创建新的tmux会话
tmux new -s training

# 3. 在tmux中启动训练
cd /home/zhanghaochen/Desktop/cours/TDSI/pjt_1/dataset/scripts
jupyter lab train_unet.ipynb
# 或者转换为Python脚本运行：
# jupyter nbconvert --to script train_unet.ipynb
# python train_unet.py

# 4. 分离tmux会话（训练继续运行）
# 按 Ctrl+B，然后按 D

# 5. 重新连接到会话
tmux attach -t training

# 6. 关闭会话
tmux kill-session -t training
```

#### 方法B: 使用nohup

```bash
# 1. 将notebook转换为Python脚本
jupyter nbconvert --to script scripts/train_unet.ipynb

# 2. 后台运行
nohup python scripts/train_unet.py > training.log 2>&1 &

# 3. 查看进程
ps aux | grep train_unet

# 4. 查看日志
tail -f training.log
```

---

## 实时监控配置

### Wandb项目组织

训练数据会自动组织为以下结构：

```
你的Wandb账号
└── medical-segmentation-unet (项目)
    └── unet-2d-training (运行)
        ├── 概览 (Overview)
        ├── 图表 (Charts)
        ├── 系统指标 (System)
        ├── 模型 (Model)
        └── 文件 (Files)
```

### 自定义监控内容

如果你想添加更多监控指标，可以在训练循环中添加：

```python
# 在训练循环中添加自定义日志
if USE_WANDB:
    wandb.log({
        'custom_metric': your_value,
        'gpu_memory_used': torch.cuda.memory_allocated() / 1e9,  # GB
        'epoch_time': epoch_duration,
    })
```

### 关闭实时监控

如果你想暂时关闭监控（比如快速测试），只需修改配置：

```python
USE_WANDB = False  # 关闭实时监控
```

训练仍会正常进行，只是不会上传数据到云端。

---

## 监控面板使用

### 主要功能区

#### 1. 概览页面 (Overview)

显示训练的基本信息：
- 运行状态（运行中/已完成/失败）
- 配置参数
- 运行时长
- 最佳性能指标

#### 2. 图表页面 (Charts)

实时显示训练指标：

**训练损失 (train_loss)**
- X轴：epoch
- Y轴：BCE损失值
- 趋势：应该逐渐下降

**验证Dice (val_dice)**
- X轴：epoch
- Y轴：Dice系数（0-1）
- 趋势：应该逐渐上升
- 目标：>0.80表示良好，>0.90表示优秀

**学习率 (learning_rate)**
- 跟踪学习率变化
- 如果使用学习率调度器，会看到阶梯状下降

**批次损失 (batch_loss)**
- 更细粒度的损失变化
- 用于检测训练不稳定性

#### 3. 预测样本 (Predictions)

每N个epoch显示一次：
- 左图：原始CT图像
- 中图：真实分割标注（红色）
- 右图：模型预测结果（蓝色）

通过对比可以直观看到模型效果的改善。

#### 4. 系统指标 (System)

监控服务器资源使用：
- GPU使用率
- GPU内存占用
- CPU使用率
- 系统内存使用

如果GPU使用率<80%，可能存在数据加载瓶颈。

#### 5. 模型图表 (Model)

如果启用了 `wandb.watch()`：
- 梯度分布
- 参数分布
- 梯度范数

用于诊断梯度消失/爆炸问题。

### 面板操作技巧

#### 自定义图表

1. 点击右上角的 "Add panel"
2. 选择 "Line plot" 或其他类型
3. 选择要显示的指标
4. 调整图表样式和布局

#### 比较多次运行

1. 在左侧勾选多个运行
2. 图表会自动叠加显示
3. 用于比较不同超参数的效果

#### 下载数据

1. 点击 "Files" 标签
2. 可以下载所有日志数据（CSV格式）
3. 可以下载保存的图像

#### 分享结果

1. 点击 "Share" 按钮
2. 获取公开链接
3. 可以分享给团队成员或导师

---

## 常见问题

### Q1: wandb登录失败怎么办？

**问题**: 运行 `wandb login` 时提示错误

**解决方案**:
```bash
# 1. 手动设置API Key
export WANDB_API_KEY=your_api_key_here

# 2. 或者在Python代码中设置
import wandb
wandb.login(key='your_api_key_here')

# 3. 如果网络受限，使用国内镜像
export WANDB_BASE_URL=https://api.wandb.ai
```

### Q2: 实时监控数据不更新？

**可能原因**:
1. 网络连接问题
2. wandb进程崩溃
3. API Key过期

**解决方案**:
```bash
# 1. 检查网络连接
ping wandb.ai

# 2. 重新登录
wandb login --relogin

# 3. 查看wandb日志
cat ~/.wandb/debug.log
```

### Q3: 训练过程中如何暂停监控？

**方案**:
```python
# 在notebook中执行
wandb.finish()  # 关闭当前运行

# 如果需要继续，可以重新初始化
wandb.init(project="...", resume=True, id=previous_run_id)
```

### Q4: 显存不足（CUDA out of memory）

**解决方案**:
```python
# 方法1: 减小batch size
BATCH_SIZE = 4  # 或更小

# 方法2: 减小图像尺寸
TARGET_SHAPE = (128, 128)

# 方法3: 减小模型规模
features = [16, 32, 64, 128]  # 原来是 [32, 64, 128, 256]

# 方法4: 使用梯度累积
# 在训练循环中每N步更新一次参数
```

### Q5: 训练速度太慢？

**优化建议**:
```python
# 1. 增加数据加载workers
num_workers = 4  # 在DataLoader中设置

# 2. 启用pin_memory
pin_memory = True

# 3. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Q6: 如何恢复中断的训练？

**方案**:
```python
# 1. 加载最新的检查点
checkpoint = torch.load('outputs/checkpoint_epoch10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# 2. 恢复wandb运行
wandb.init(project="...", resume=True, id=previous_run_id)

# 3. 从start_epoch继续训练
for epoch in range(start_epoch, EPOCHS + 1):
    ...
```

### Q7: 验证Dice系数很低（<0.5）？

**可能原因和解决方案**:

1. **训练轮数不够**: 增加到50-100个epoch
2. **学习率不合适**: 尝试1e-4或5e-4
3. **数据预处理问题**: 检查归一化是否正确
4. **模型容量不够**: 增加features列表的值
5. **数据不平衡**: 考虑使用Dice Loss或Focal Loss

### Q8: 如何在手机上查看训练进度？

**步骤**:
1. 在手机浏览器中访问 wandb.ai
2. 登录你的账号
3. 选择对应的项目和运行
4. 所有图表都会自适应手机屏幕

**提示**: 可以将wandb链接添加到手机主屏幕，方便快速访问。

---

## 最佳实践

### 1. 训练前的准备清单

- [ ] 确认数据路径正确
- [ ] wandb已登录并测试
- [ ] 配置参数已根据GPU调整
- [ ] 输出目录有足够空间
- [ ] 使用tmux或nohup防止中断
- [ ] 记录运行配置和目标

### 2. 训练中的监控要点

#### 每小时检查：
- 训练损失是否持续下降
- GPU使用率是否正常（>70%）
- 没有出现错误或警告

#### 每天检查：
- 验证Dice是否提升
- 查看预测样本质量
- 对比历史最佳性能
- 评估是否需要早停

### 3. 训练结束后的分析

#### 性能评估
```python
# 查看最佳模型的性能
print(f"最佳验证Dice: {max(history['val_dice']):.4f}")
print(f"最佳epoch: {history['val_dice'].index(max(history['val_dice'])) + 1}")

# 在wandb面板中对比多次运行
# 选择最佳超参数组合
```

#### 结果保存
```bash
# 1. 下载wandb数据
wandb export your-username/medical-segmentation-unet/run-id

# 2. 备份检查点
cp outputs/checkpoint_epoch*.pth /backup/location/

# 3. 保存训练报告
# 在wandb面板点击"Export" -> "PDF Report"
```

### 4. 超参数调优建议

基于实时监控结果，逐步优化超参数：

#### 第一轮：快速验证（2-5 epochs）
```python
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
```
目标：确认代码运行无误，损失正常下降

#### 第二轮：中等训练（20-30 epochs）
```python
EPOCHS = 30
BATCH_SIZE = 16  # 如果GPU足够
LEARNING_RATE = 1e-3
```
目标：评估模型收敛趋势，调整学习率

#### 第三轮：完整训练（50-100 epochs）
```python
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 5e-4  # 根据第二轮结果调整
# 添加学习率调度器
```
目标：获得最佳性能模型

### 5. 团队协作建议

#### 共享项目
```python
# 在wandb配置中设置团队项目
WANDB_PROJECT = 'team-name/medical-segmentation'
```

#### 命名规范
```python
# 使用描述性的运行名称
import datetime
WANDB_RUN_NAME = f"unet-bs{BATCH_SIZE}-lr{LEARNING_RATE}-{datetime.now().strftime('%m%d')}"
```

#### 标注实验
```python
WANDB_NOTES = """
实验目标: 测试更大的batch size
变更内容: BATCH_SIZE 8 -> 16
预期结果: 训练更稳定，Dice提升
"""
```

### 6. 故障恢复策略

#### 自动保存检查点
```python
# 每N个epoch保存一次
if epoch % 5 == 0:
    torch.save({...}, f'checkpoint_epoch{epoch}.pth')
```

#### 定期备份
```bash
# 设置定时任务
crontab -e

# 每6小时备份一次
0 */6 * * * rsync -av /local/hzhang02/data/dataset/outputs/ /backup/location/
```

#### 监控脚本
```bash
# 创建监控脚本 monitor.sh
#!/bin/bash
while true; do
    if ! pgrep -f "train_unet" > /dev/null; then
        echo "Training process died at $(date)" >> training_monitor.log
        # 可以添加邮件通知
    fi
    sleep 300  # 每5分钟检查一次
done
```

---

## 项目文件结构

```
dataset/
├── README_1121.md                    # 本文件（实时监控版使用指南）
├── README_1114.md                    # 旧版README
├── README_TRAIN_1114.md              # 旧版训练说明
├── 实时检测的办法.md                  # 实时监控技术文档
├── MODEL_GUIDE_CHINESE_1114.md       # 模型原理指南（中文）
├── MODEL_GUIDE_FRENCH_1114.md        # 模型原理指南（法文）
├── scripts/
│   ├── train_unet.ipynb              # 训练notebook（已集成wandb）
│   ├── segmentation_detection_analysis.ipynb  # 分析notebook
│   ├── README_1114.md                # Scripts目录说明
│   ├── README_CHINESE_1114.md        # Scripts说明（中文）
│   └── README_FRENCH_1114.md         # Scripts说明（法文）
├── outputs/                          # 训练输出目录
│   ├── checkpoint_epoch*.pth         # 模型检查点
│   ├── training_history.json         # 训练历史
│   ├── label_map.json                # 标签映射
│   └── *.png                         # 可视化结果
├── s0000/, s0001/, ...               # 数据集
│   ├── ct.nii.gz                     # CT扫描
│   └── segmentations/                # 分割标注
└── wandb/                            # Wandb本地缓存（自动生成）
```

---

## 技术支持

### 获取帮助

1. **查看文档**:
   - `实时检测的办法.md`: 详细的监控技术说明
   - `MODEL_GUIDE_CHINESE_1114.md`: 模型原理和实现细节

2. **在线资源**:
   - Wandb官方文档: https://docs.wandb.ai/
   - PyTorch文档: https://pytorch.org/docs/
   - U-Net论文: https://arxiv.org/abs/1505.04597

3. **社区支持**:
   - Wandb Community: https://community.wandb.ai/
   - PyTorch Forums: https://discuss.pytorch.org/

### 报告问题

如果遇到问题，请提供以下信息：
- 错误信息和完整堆栈跟踪
- 训练配置参数
- GPU型号和显存大小
- Wandb运行链接（如果可用）
- Python和PyTorch版本

---

## 更新日志

### 版本 1121 (2025-11-21)
- ✨ 新增：集成Weights & Biases实时监控
- ✨ 新增：训练样本可视化
- ✨ 新增：自动记录最佳模型
- 📝 文档：新增完整的云服务器使用指南
- 📝 文档：新增常见问题和最佳实践

### 版本 1114 (2025-11-14)
- 初始版本：基础U-Net训练流程
- 支持117个解剖结构分割
- 基本的本地训练和评估

---

## 许可和引用

本项目基于以下开源工具和数据集：

- **模型架构**: U-Net (Ronneberger et al., 2015)
- **深度学习框架**: PyTorch
- **实时监控**: Weights & Biases
- **数据集**: TotalSegmentator (根据你的实际数据集调整)

如果本项目对你的研究有帮助，请考虑引用相关论文。

---

## 联系方式

- **项目维护**: Claude Code
- **创建日期**: 2025年11月21日
- **最后更新**: 2025年11月21日

---

**祝训练顺利！如有问题，请查看wandb仪表盘或参考本文档的常见问题部分。** 🚀
