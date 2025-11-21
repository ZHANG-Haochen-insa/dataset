# 3D医学图像分割 - 训练与分析系统

本目录包含用于训练和分析CT扫描3D医学图像分割模型的Jupyter notebook。

## 📋 目录内容

- `train_unet.ipynb` - U-Net模型训练notebook
- `segmentation_detection_analysis.ipynb` - 模型评估与可视化notebook

## 🎯 项目概述

本项目实现了基于2D U-Net的多器官分割模型，可从CT扫描中分割117种不同的解剖结构，包括：
- 主要器官（大脑、心脏、肝脏、肾脏、肺部等）
- 骨骼结构（椎骨、肋骨等）
- 血管系统（主动脉、动脉、静脉等）
- 肌肉群

## 🔧 系统要求

### 硬件要求
- Python 3.10+
- 支持CUDA的GPU（推荐）或CPU
- 8GB以上内存（推荐16GB以上）

### Python依赖
所有依赖项列在 `../requirements.txt` 中：
```
numpy          # 数值计算
nibabel        # NIfTI文件读写
scikit-image   # 图像处理
torch          # 深度学习框架
matplotlib     # 2D可视化
plotly         # 3D交互式可视化
pandas         # 数据分析
tqdm           # 进度条
scipy          # 科学计算
jupyter        # Jupyter notebook
jupyterlab     # JupyterLab环境
ipywidgets     # 交互式组件
```

## 📦 安装步骤

1. **安装依赖** （如果尚未安装）：
```bash
cd /home/hzhang02/dataset
pip3 install -r requirements.txt
```

2. **配置Jupyter内核**：
```bash
python3 -m ipykernel install --user --name=claude_env --display-name="Python 3 (claude_env)"
```

3. **验证安装**：
```bash
python3 -c "import numpy, torch, nibabel, plotly; print('所有包已成功安装！')"
```

## 🚀 快速开始

### 第一步：训练模型

1. 启动JupyterLab：
```bash
cd /home/hzhang02/dataset/scripts
jupyter lab train_unet.ipynb
```

2. 选择内核：**"Python 3 (claude_env)"**

3. 运行所有单元格以完成：
   - 加载和准备数据集
   - 构建U-Net模型
   - 训练指定轮数
   - 保存检查点到 `../outputs/`

**训练配置参数** （可在第6节调整）：
- `EPOCHS = 5` - 训练轮数
- `BATCH_SIZE = 8` - 批次大小
- `LEARNING_RATE = 1e-3` - 学习率
- `TARGET_SHAPE = (256, 256)` - 图像尺寸

**预期输出**：
- `outputs/label_map.json` - 解剖结构到通道的映射
- `outputs/checkpoint_epochX.pth` - 模型检查点
- `outputs/training_history.json` - 训练指标
- `outputs/training_history.png` - 训练曲线可视化

### 第二步：评估与分析

1. 启动分析notebook：
```bash
jupyter lab segmentation_detection_analysis.ipynb
```

2. 选择内核：**"Python 3 (claude_env)"**

3. 运行所有单元格以完成：
   - 加载训练好的模型
   - 对测试数据进行推理
   - 计算评估指标
   - 生成可视化结果

**评估指标**：
- Dice系数
- IoU（交并比）
- 检测成功率（不同阈值）
- 每个结构的性能分析

**生成的输出**：
- `outputs/evaluation_results_*.csv` - 每个结构的详细指标
- `outputs/metrics_distribution_*.png` - 指标分布图
- `outputs/structure_ranking_*.png` - 性能排名图
- `outputs/segmentation_visualization_*.png` - 2D分割结果
- `outputs/3d_*.html` - 交互式3D可视化

## 📊 数据集结构

预期的目录结构：
```
/home/hzhang02/dataset/
├── s0000/
│   ├── ct.nii.gz                    # CT扫描图像
│   └── segmentations/               # 真实标注掩码
│       ├── liver.nii.gz             # 肝脏
│       ├── heart.nii.gz             # 心脏
│       ├── kidney_left.nii.gz       # 左肾
│       └── ... (共117个结构)
├── s0001/
├── s0002/
└── ...
```

## 📈 典型工作流程

1. **初步训练** （2-5轮用于测试）：
```bash
# 运行 train_unet.ipynb，设置 EPOCHS=2
```

2. **快速评估**：
```bash
# 运行 segmentation_detection_analysis.ipynb
```

3. **完整训练** （如果结果理想）：
```bash
# 在 train_unet.ipynb 中增加 EPOCHS 至 20-50
```

4. **全面分析**：
```bash
# 使用最佳检查点重新运行 segmentation_detection_analysis.ipynb
```

## 🎨 可视化示例

notebook会生成各种可视化结果：

### 训练进度
- 各轮次的损失曲线
- 验证集Dice得分趋势
- 样本预测与真实标注对比

### 评估结果
- 2D切片对比（CT + 叠加）
- 3D网格渲染（交互式HTML）
- 性能分布直方图
- 结构排名图表

## 🔍 常见问题解决

### 问题："ModuleNotFoundError: No module named 'numpy'"
**解决方案**：确保选择了正确的内核：
- 在Jupyter中：Kernel → Change Kernel → "Python 3 (claude_env)"

### 问题：CUDA内存不足
**解决方案**：减少训练notebook中的批次大小：
```python
BATCH_SIZE = 4  # 甚至可以设为2
```

### 问题：训练速度太慢
**解决方案**：
- 减小图像尺寸：`TARGET_SHAPE = (128, 128)`
- 降低模型复杂度：`features=[16, 32, 64, 128]`
- 减少训练轮数进行测试：`EPOCHS = 2`

### 问题：Dice得分偏低
**可能原因**：
- 训练轮数不足（尝试20-50轮）
- 学习率过高/过低（尝试1e-4或5e-4）
- 数据集太小（考虑数据增强）

## 📚 模型架构

**U-Net 2D**：
- 编码器：4个下采样块 [32, 64, 128, 256 特征]
- 瓶颈层：512 特征
- 解码器：4个上采样块（带跳跃连接）
- 输出层：117通道（每个解剖结构一个）

**损失函数**：带Logits的二元交叉熵（BCEWithLogitsLoss）

**优化器**：Adam (lr=1e-3)

**评估指标**：Dice系数

## 💡 获得更好结果的建议

1. **数据增强**：考虑添加：
   - 随机旋转（±15°）
   - 随机翻转（水平/垂直）
   - 弹性变形
   - 强度缩放

2. **高级训练技巧**：
   - 学习率调度（ReduceLROnPlateau）
   - 早停（Early Stopping）
   - 梯度裁剪
   - 混合精度训练（加速GPU训练）

3. **模型改进**：
   - 尝试3D U-Net替代2D
   - 使用注意力机制（Attention U-Net）
   - 尝试不同损失函数（Dice Loss、Focal Loss）

4. **集成方法**：
   - 使用不同随机种子训练多个模型
   - 对预测结果取平均以获得更好效果

## 📖 参考资料

- **数据集**：TotalSegmentator v2.0.1
- **模型**：U-Net (Ronneberger et al., 2015)
- **框架**：PyTorch 2.9.0

## 📝 使用说明

- 训练时间取决于GPU/CPU和数据集大小
- 第一轮通常较慢，因为需要加载数据
- 每轮训练后会保存检查点（文件可能较大）
- 每轮训练后会进行验证

## 🎓 学习资源

### 推荐阅读
1. **U-Net论文**：Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. **医学图像分割综述**：深入了解医学图像处理技术
3. **PyTorch官方教程**：学习深度学习框架基础

### 相关工具
- **3D Slicer**：医学图像可视化和分析
- **ITK-SNAP**：医学图像分割工具
- **MONAI**：医学成像AI框架

## 🤝 项目扩展

要扩展此项目，可以：
1. 添加新的解剖结构到标签映射
2. 实现3D U-Net以获得更好的空间上下文
3. 添加更多评估指标（Hausdorff距离、表面距离）
4. 与医学影像查看器集成（3D Slicer、ITK-SNAP）
5. 实现实时推理API
6. 添加模型可解释性分析

## ⚙️ 高级配置

### GPU内存优化
```python
# 在训练notebook中添加
torch.cuda.empty_cache()  # 清理GPU缓存
torch.backends.cudnn.benchmark = True  # 加速训练
```

### 数据增强示例
```python
# 可以在数据集类中添加
import torchvision.transforms as T
transform = T.Compose([
    T.RandomRotation(15),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
])
```

### 学习率调度
```python
# 在训练循环中添加
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
# 在每个epoch后调用
scheduler.step(val_dice)
```

## 📊 性能基准

典型性能指标（参考）：
- **训练时间**：约2-5分钟/epoch（GPU）或20-30分钟/epoch（CPU）
- **推理速度**：约0.1-0.5秒/切片（GPU）
- **内存占用**：4-8GB GPU内存（batch_size=8）
- **Dice得分**：0.6-0.9（取决于结构和训练轮数）

## 🐛 调试技巧

1. **检查数据加载**：
```python
# 在notebook中运行
sample_img, sample_mask = train_ds[0]
print(f"图像形状: {sample_img.shape}")
print(f"掩码形状: {sample_mask.shape}")
print(f"掩码范围: {sample_mask.min()}-{sample_mask.max()}")
```

2. **验证模型输出**：
```python
# 检查模型输出维度
with torch.no_grad():
    test_output = model(sample_img.unsqueeze(0).to(device))
    print(f"输出形状: {test_output.shape}")
```

3. **监控GPU使用**：
```bash
# 在终端运行
nvidia-smi -l 1  # 每秒更新一次GPU状态
```

## 📄 许可证

本项目仅供教育和研究使用。

## 🙏 致谢

- TotalSegmentator数据集提供者
- PyTorch团队
- 医学图像处理社区

---

**创建时间**：2025年11月
**最后更新**：2025年11月
**维护者**：Claude Code
**版本**：1.0.0

---

## 💬 常见问题 FAQ

**Q: 可以在CPU上训练吗？**
A: 可以，但速度会很慢。建议减小batch_size和图像尺寸。

**Q: 训练需要多长时间？**
A: 取决于硬件。GPU上约2-5分钟/epoch，CPU上约20-30分钟/epoch。

**Q: 如何保存最佳模型？**
A: 在训练循环中添加逻辑，保存验证Dice最高的模型。

**Q: 可以用于其他医学图像吗？**
A: 可以，但需要相应的标注数据和可能的模型调整。

**Q: 如何提高分割精度？**
A: 增加训练轮数、使用数据增强、尝试3D U-Net、调整超参数。

## 🔗 相关链接

- [PyTorch官网](https://pytorch.org/)
- [TotalSegmentator项目](https://github.com/wasserth/TotalSegmentator)
- [医学图像分割资源](https://grand-challenge.org/)

---

**祝你训练顺利！如有问题，请查看文档或检查常见问题部分。** 🚀
