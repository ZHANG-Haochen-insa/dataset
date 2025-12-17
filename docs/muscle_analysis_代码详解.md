# muscle_analysis.py 代码详解

**文档目的**: 逐行解释肌肉量分析脚本的运行逻辑，帮助理解深度学习推理流程
**适用对象**: Python和深度学习初学者
**代码位置**: `scripts/muscle_analysis.py`

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [导入库详解](#2-导入库详解)
3. [U-Net模型结构](#3-u-net模型结构)
4. [MuscleAnalyzer类详解](#4-muscleanalyzer类详解)
5. [主函数流程](#5-主函数流程)
6. [完整执行流程图](#6-完整执行流程图)

---

## 1. 整体架构概览

```
muscle_analysis.py
│
├── 导入库 (第21-33行)
│
├── 模型定义 (第40-109行)
│   ├── DoubleConv类: 双卷积块
│   └── UNet2D类: U-Net网络
│
├── 常量定义 (第117-142行)
│   ├── MUSCLE_NAMES: 肌肉名称字典
│   └── MUSCLE_COLORS: 可视化颜色
│
├── MuscleAnalyzer类 (第149-521行)
│   ├── __init__: 初始化（加载模型）
│   ├── preprocess_slice: 预处理切片
│   ├── predict_slice: 预测单个切片
│   ├── analyze_ct: 分析整个CT
│   ├── _save_results: 保存结果
│   ├── _generate_visualizations: 生成可视化
│   └── print_report: 打印报告
│
└── main函数 (第528-569行): 程序入口
```

---

## 2. 导入库详解

```python
#!/usr/bin/env python           # 第1行: shebang，告诉系统用python执行
# -*- coding: utf-8 -*-         # 第2行: 声明文件编码为UTF-8，支持中文
```

### 2.1 标准库

```python
import os                       # 第21行: 操作系统接口，用于文件路径操作
import json                     # 第22行: JSON文件读写
import argparse                 # 第23行: 命令行参数解析
```

**用途示例**:
```python
os.path.join('a', 'b')         # 拼接路径: 'a/b'
os.makedirs('dir', exist_ok=True)  # 创建目录，已存在不报错
json.load(f)                   # 从文件读取JSON
json.dump(data, f)             # 写入JSON到文件
```

### 2.2 科学计算库

```python
import numpy as np              # 第24行: 数值计算核心库
import nibabel as nib           # 第25行: 医学图像格式(NIfTI)读写
from skimage.transform import resize  # 第26行: 图像缩放
```

**numpy关键操作**:
```python
np.zeros((10, 256, 256))       # 创建全零数组
np.percentile(arr, 99)          # 计算第99百分位数
np.clip(arr, 0, 1)              # 将数组值限制在[0,1]范围
arr.sum()                       # 求和
arr.astype(np.float32)          # 类型转换
```

**nibabel关键操作**:
```python
nii = nib.load('ct.nii.gz')    # 加载NIfTI文件
data = nii.get_fdata()          # 获取图像数据(3D数组)
header = nii.header             # 获取头信息(包含体素尺寸等)
affine = nii.affine             # 获取仿射变换矩阵(空间坐标)
nib.save(nii, 'output.nii.gz')  # 保存NIfTI文件
```

### 2.3 深度学习库

```python
import torch                    # 第27行: PyTorch核心
import torch.nn as nn           # 第28行: 神经网络模块
```

**torch关键操作**:
```python
torch.cuda.is_available()       # 检查GPU是否可用
torch.device('cuda')            # 指定设备
torch.load('model.pth')         # 加载模型权重
torch.no_grad()                 # 禁用梯度计算(推理时使用)
torch.sigmoid(x)                # Sigmoid激活函数
tensor.to(device)               # 将张量移到指定设备
tensor.cpu().numpy()            # 转为numpy数组
```

### 2.4 可视化和数据处理

```python
import matplotlib.pyplot as plt          # 第29行: 绑图库
from matplotlib.colors import ListedColormap  # 第30行: 自定义颜色映射
import pandas as pd                      # 第31行: 数据表格处理
from datetime import datetime            # 第32行: 时间处理
from tqdm import tqdm                    # 第33行: 进度条
```

---

## 3. U-Net模型结构

### 3.1 DoubleConv类 (双卷积块)

```python
class DoubleConv(nn.Module):
    """
    双卷积块，U-Net的基本构建单元
    结构: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
        """
        super().__init__()  # 调用父类构造函数

        # nn.Sequential: 按顺序执行的容器
        self.net = nn.Sequential(
            # 第一个卷积: 3x3卷积，padding=1保持尺寸不变
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # 批归一化: 加速训练，稳定梯度
            nn.BatchNorm2d(out_ch),
            # ReLU激活: max(0, x)，引入非线性
            nn.ReLU(inplace=True),  # inplace=True节省内存

            # 第二个卷积
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """前向传播"""
        return self.net(x)
```

**图解**:
```
输入 (batch, in_ch, H, W)
    ↓
Conv2d(3x3) → (batch, out_ch, H, W)
    ↓
BatchNorm2d
    ↓
ReLU
    ↓
Conv2d(3x3) → (batch, out_ch, H, W)
    ↓
BatchNorm2d
    ↓
ReLU
    ↓
输出 (batch, out_ch, H, W)
```

### 3.2 UNet2D类

```python
class UNet2D(nn.Module):
    """
    2D U-Net: 编码器-解码器结构，带跳跃连接
    """

    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128, 256]):
        """
        Args:
            in_ch: 输入通道数 (灰度图=1, RGB=3)
            out_ch: 输出通道数 (类别数)
            features: 每层的特征通道数
        """
        super().__init__()

        # ModuleList: 可以存储多个子模块的列表
        self.downs = nn.ModuleList()  # 编码器（下采样）模块
        self.ups = nn.ModuleList()    # 解码器（上采样）模块
```

#### 编码器构建

```python
        # 构建编码器
        # features = [32, 64, 128, 256]
        for f in features:
            # 每层: DoubleConv + MaxPool
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f  # 下一层的输入通道 = 当前层输出通道

        # 最大池化: 尺寸减半
        self.pool = nn.MaxPool2d(2)
```

**编码器结构**:
```
输入: (1, 256, 256)      # 1通道, 256x256
    ↓ DoubleConv
(32, 256, 256)
    ↓ MaxPool2d(2)
(32, 128, 128)           # 尺寸减半
    ↓ DoubleConv
(64, 128, 128)
    ↓ MaxPool2d(2)
(64, 64, 64)
    ↓ DoubleConv
(128, 64, 64)
    ↓ MaxPool2d(2)
(128, 32, 32)
    ↓ DoubleConv
(256, 32, 32)
    ↓ MaxPool2d(2)
(256, 16, 16)
```

#### 瓶颈层

```python
        # 瓶颈层: 最深层，通道数翻倍
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # 256 -> 512
```

**瓶颈**:
```
(256, 16, 16)
    ↓ DoubleConv
(512, 16, 16)            # 特征最丰富的层
```

#### 解码器构建

```python
        # 构建解码器
        rev = list(reversed(features))  # [256, 128, 64, 32]
        up_in = features[-1] * 2        # 512

        for f in rev:
            # 转置卷积: 上采样，尺寸翻倍
            self.ups.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            # DoubleConv: 处理拼接后的特征
            # 输入通道 = f(上采样) + f(跳跃连接) = 2f = up_in
            self.ups.append(DoubleConv(up_in, f))
            up_in = f
```

**解码器结构**:
```
(512, 16, 16)
    ↓ ConvTranspose2d(2x2)
(256, 32, 32)
    + 跳跃连接 (256, 32, 32)
    ↓ 拼接
(512, 32, 32)
    ↓ DoubleConv
(256, 32, 32)
    ↓ ...继续上采样...
(32, 256, 256)
```

#### 最终输出层

```python
        # 1x1卷积: 将特征映射到类别数
        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)
        # 32 -> 10 (10种肌肉)
```

#### 前向传播

```python
    def forward(self, x):
        """
        前向传播过程

        Args:
            x: 输入图像 (batch, 1, 256, 256)
        Returns:
            分割结果 (batch, num_classes, 256, 256)
        """
        # ========== 编码路径 ==========
        skips = []  # 存储跳跃连接的特征

        for down in self.downs:
            x = down(x)           # 双卷积
            skips.append(x)       # 保存用于跳跃连接
            x = self.pool(x)      # 下采样

        # ========== 瓶颈 ==========
        x = self.bottleneck(x)

        # ========== 解码路径 ==========
        # self.ups = [Trans, Conv, Trans, Conv, ...]
        # 每次取2个: 转置卷积 + 双卷积
        for idx in range(0, len(self.ups), 2):
            trans = self.ups[idx]      # 转置卷积
            conv = self.ups[idx + 1]   # 双卷积

            x = trans(x)               # 上采样

            # 获取对应的跳跃连接
            # idx=0 -> skips[-1], idx=2 -> skips[-2], ...
            skip = skips[-(idx // 2) - 1]

            # 处理尺寸不匹配（边缘情况）
            if x.shape != skip.shape:
                _, _, h, w = x.shape
                skip = skip[:, :, :h, :w]  # 裁剪skip

            # 拼接: 沿通道维度
            x = torch.cat([skip, x], dim=1)

            x = conv(x)  # 双卷积处理拼接后的特征

        # ========== 输出 ==========
        return self.final(x)  # 1x1卷积，得到类别预测
```

**完整U-Net数据流图**:
```
输入 (1, 256, 256)
       │
       ↓ DoubleConv
    ┌──(32, 256, 256)──────────────────────────────────┐
    │  ↓ Pool                                          │ 跳跃连接
    │  (32, 128, 128)                                  │
    │  ↓ DoubleConv                                    │
    │┌─(64, 128, 128)─────────────────────────┐        │
    ││ ↓ Pool                                 │        │
    ││ (64, 64, 64)                           │        │
    ││ ↓ DoubleConv                           │        │
    ││┌(128, 64, 64)──────────────┐           │        │
    │││↓ Pool                     │           │        │
    │││(128, 32, 32)              │           │        │
    │││↓ DoubleConv               │           │        │
    │││┌(256, 32, 32)────┐        │           │        │
    ││││↓ Pool           │        │           │        │
    ││││(256, 16, 16)    │        │           │        │
    ││││↓ Bottleneck     │        │           │        │
    ││││(512, 16, 16)    │        │           │        │
    ││││↓ TransConv      │        │           │        │
    ││││(256, 32, 32)    │        │           │        │
    │││└───→ Cat ────────┘        │           │        │
    │││    (512, 32, 32)          │           │        │
    │││    ↓ DoubleConv           │           │        │
    │││    (256, 32, 32)          │           │        │
    │││    ↓ TransConv            │           │        │
    │││    (128, 64, 64)          │           │        │
    ││└────────→ Cat ─────────────┘           │        │
    ││         (256, 64, 64)                  │        │
    ││         ↓ DoubleConv                   │        │
    ││         (128, 64, 64)                  │        │
    ││         ↓ TransConv                    │        │
    ││         (64, 128, 128)                 │        │
    │└─────────────→ Cat ─────────────────────┘        │
    │              (128, 128, 128)                     │
    │              ↓ DoubleConv                        │
    │              (64, 128, 128)                      │
    │              ↓ TransConv                         │
    │              (32, 256, 256)                      │
    └──────────────────→ Cat ──────────────────────────┘
                       (64, 256, 256)
                       ↓ DoubleConv
                       (32, 256, 256)
                       ↓ Final Conv(1x1)
                       (10, 256, 256)  ← 输出: 10种肌肉的预测
```

---

## 4. MuscleAnalyzer类详解

### 4.1 初始化 `__init__`

```python
class MuscleAnalyzer:
    def __init__(self, model_path, label_map_path, device=None):
        """
        初始化分析器

        这个方法在创建MuscleAnalyzer对象时自动调用，
        完成模型加载和设备设置等准备工作。
        """

        # ========== 第1步: 设置计算设备 ==========
        if device is None:
            # 自动检测: 有GPU用GPU，否则用CPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # 使用用户指定的设备
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        # ========== 第2步: 加载标签映射 ==========
        # label_map.json内容示例:
        # {"autochthon_left.nii.gz": 0, "autochthon_right.nii.gz": 1, ...}
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)

        self.num_classes = len(self.label_map)  # 类别数量 = 10
        print(f"加载标签映射: {self.num_classes} 种肌肉")

        # ========== 第3步: 创建反向映射 ==========
        # 反向映射: 索引 -> 肌肉名称
        # {0: 'autochthon_left', 1: 'autochthon_right', ...}
        self.idx_to_label = {v: k.replace('.nii.gz', '') for k, v in self.label_map.items()}

        # ========== 第4步: 加载模型 ==========
        # 4.1 创建模型实例（结构必须与训练时一致）
        self.model = UNet2D(
            in_ch=1,           # 输入: 灰度图
            out_ch=self.num_classes,  # 输出: 10种肌肉
            features=[32, 64, 128, 256]
        )

        # 4.2 加载权重
        # checkpoint包含: model_state_dict, optimizer_state_dict, epoch等
        checkpoint = torch.load(model_path, map_location=self.device)

        # 4.3 将权重加载到模型
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 4.4 将模型移到指定设备
        self.model.to(self.device)

        # 4.5 设置为评估模式
        # 重要! 这会禁用Dropout和改变BatchNorm行为
        self.model.eval()

        print(f"模型加载成功: {model_path}")

        # ========== 第5步: 设置目标尺寸 ==========
        # 模型训练时使用的输入尺寸
        self.target_shape = (256, 256)
```

### 4.2 预处理切片 `preprocess_slice`

```python
    def preprocess_slice(self, slice_img):
        """
        预处理单个CT切片

        为什么需要预处理？
        1. CT值范围很大（-1000到3000+），需要归一化
        2. 原始图像尺寸可能不是256x256，需要调整
        3. 需要转换为PyTorch张量格式

        Args:
            slice_img: 原始CT切片，numpy数组 (H, W)

        Returns:
            预处理后的tensor (1, 1, H, W)
            第一个1是batch维度，第二个1是通道维度
        """

        # ========== 第1步: 百分位数窗口化 ==========
        # 为什么用百分位数？避免异常值（如金属伪影）影响归一化
        lo = np.percentile(slice_img, 1)   # 第1百分位数
        hi = np.percentile(slice_img, 99)  # 第99百分位数

        # clip: 将值限制在[lo, hi]范围内
        # 低于lo的设为lo，高于hi的设为hi
        slice_img = np.clip(slice_img, lo, hi)

        # ========== 第2步: 归一化到[0,1] ==========
        if hi - lo > 0:
            # Min-Max归一化
            slice_img = (slice_img - lo) / (hi - lo)
        else:
            # 特殊情况: 整个切片值相同，返回全零
            slice_img = np.zeros_like(slice_img)

        # ========== 第3步: 调整尺寸 ==========
        # resize参数:
        # - order=1: 双线性插值（平滑）
        # - preserve_range=True: 保持值范围[0,1]
        # - anti_aliasing=True: 抗锯齿
        slice_img = resize(
            slice_img,
            self.target_shape,  # (256, 256)
            order=1,
            preserve_range=True,
            anti_aliasing=True
        )

        # ========== 第4步: 转换为PyTorch张量 ==========
        # numpy (256, 256)
        # -> torch (256, 256)
        # -> unsqueeze(0) (1, 256, 256)  添加通道维度
        # -> unsqueeze(0) (1, 1, 256, 256)  添加batch维度
        img_t = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float()

        return img_t
```

**数据变换过程**:
```
原始CT切片: (261, 182), 值范围[-1000, 2000]
    ↓ 百分位数裁剪
值范围[lo, hi]
    ↓ 归一化
值范围[0, 1]
    ↓ resize
(256, 256), 值范围[0, 1]
    ↓ 转tensor + 添加维度
torch.Tensor (1, 1, 256, 256)
```

### 4.3 预测切片 `predict_slice`

```python
    def predict_slice(self, slice_tensor):
        """
        对单个切片进行预测

        Args:
            slice_tensor: 预处理后的tensor (1, 1, 256, 256)

        Returns:
            预测掩码 numpy数组 (num_classes, 256, 256)
            每个通道是一种肌肉的二值掩码
        """

        # ========== 关键: 禁用梯度计算 ==========
        # 推理时不需要计算梯度，节省内存和时间
        with torch.no_grad():

            # 将输入移到GPU（如果可用）
            slice_tensor = slice_tensor.to(self.device)

            # 前向传播，获得原始输出（logits）
            # 输出形状: (1, 10, 256, 256)
            logits = self.model(slice_tensor)

            # Sigmoid激活: 将logits转换为概率[0,1]
            # 为什么用sigmoid而不是softmax？
            # 因为这是多标签分类，一个像素可能属于多个类别
            pred = torch.sigmoid(logits)

            # 二值化: 概率>0.5视为属于该类别
            pred_bin = (pred > 0.5).float()

        # 转回numpy，去掉batch维度
        # (1, 10, 256, 256) -> (10, 256, 256)
        return pred_bin[0].cpu().numpy()
```

**预测流程**:
```
输入: (1, 1, 256, 256)
    ↓ U-Net前向传播
logits: (1, 10, 256, 256)  # 原始输出，可正可负
    ↓ Sigmoid
概率: (1, 10, 256, 256)    # 值在[0,1]之间
    ↓ 阈值0.5
二值掩码: (1, 10, 256, 256) # 0或1
    ↓ 去batch维度
输出: (10, 256, 256)
```

### 4.4 分析整个CT `analyze_ct`

```python
    def analyze_ct(self, ct_path, output_dir=None):
        """
        分析整个CT图像，这是核心方法

        处理流程:
        1. 加载CT图像
        2. 获取体素尺寸
        3. 逐层分割
        4. 计算体积
        5. 保存结果
        """

        # ========== 第1步: 加载CT图像 ==========
        ct_nii = nib.load(ct_path)           # 加载NIfTI文件
        ct_data = ct_nii.get_fdata()         # 获取3D数组
        ct_data = ct_data.astype(np.float32) # 转为float32节省内存
        header = ct_nii.header               # 头信息
        affine = ct_nii.affine               # 仿射矩阵（空间定位用）

        # ========== 第2步: 获取体素尺寸 ==========
        # 体素 = 3D像素，有物理尺寸
        voxel_dims = header.get_zooms()  # 返回(dx, dy, dz)，单位mm

        # 切片内像素面积 = dx * dy (mm²)
        voxel_size_mm = float(voxel_dims[0]) * float(voxel_dims[1])

        # 切片厚度 = dz (mm)
        slice_thickness = float(voxel_dims[2]) if len(voxel_dims) > 2 else 1.0

        # 单个体素体积 = dx * dy * dz (mm³)
        voxel_volume_mm3 = voxel_size_mm * slice_thickness

        # ========== 第3步: 准备存储 ==========
        original_shape = ct_data.shape[:2]  # (H, W)
        depth = ct_data.shape[2]            # 切片数量

        # 预分配结果数组
        # 形状: (10, H, W, depth) - 10种肌肉，每种一个3D掩码
        all_predictions = np.zeros(
            (self.num_classes, *original_shape, depth),
            dtype=np.float32
        )

        # 每层面积记录，用于统计
        slice_areas = {i: [] for i in range(self.num_classes)}

        # ========== 第4步: 逐层分割 ==========
        for z in tqdm(range(depth), desc="处理切片"):
            # 4.1 提取第z层切片
            slice_img = ct_data[:, :, z]  # (H, W)

            # 4.2 预处理
            slice_tensor = self.preprocess_slice(slice_img)

            # 4.3 模型预测
            pred_mask = self.predict_slice(slice_tensor)
            # pred_mask形状: (10, 256, 256)

            # 4.4 将预测结果调整回原始尺寸
            for c in range(self.num_classes):
                # order=0: 最近邻插值，保持二值特性
                pred_resized = resize(
                    pred_mask[c],
                    original_shape,
                    order=0,  # 最近邻
                    preserve_range=True,
                    anti_aliasing=False  # 二值图不需要抗锯齿
                )

                # 再次二值化（resize可能引入中间值）
                all_predictions[c, :, :, z] = (pred_resized > 0.5).astype(np.float32)

                # 4.5 计算该层面积
                pixel_count = all_predictions[c, :, :, z].sum()
                area_mm2 = pixel_count * voxel_size_mm  # 像素数 × 像素面积
                slice_areas[c].append(area_mm2)

        # ========== 第5步: 计算总体积 ==========
        results = {
            'ct_path': ct_path,
            'ct_shape': list(ct_data.shape),
            'voxel_dims_mm': [float(v) for v in voxel_dims],
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'muscles': {}
        }

        total_muscle_volume = 0

        for c in range(self.num_classes):
            muscle_name = self.idx_to_label[c]

            # 5.1 计算体积
            total_pixels = all_predictions[c].sum()  # 总像素数
            volume_mm3 = total_pixels * voxel_volume_mm3  # 体积(mm³)
            volume_cm3 = volume_mm3 / 1000  # 体积(cm³)

            # 5.2 计算面积统计
            non_zero_areas = [a for a in slice_areas[c] if a > 0]
            avg_area_mm2 = np.mean(non_zero_areas) if non_zero_areas else 0
            max_area_mm2 = max(slice_areas[c]) if slice_areas[c] else 0

            # 5.3 找到肌肉出现的切片范围
            slices_with_muscle = [i for i, a in enumerate(slice_areas[c]) if a > 0]
            slice_range = [min(slices_with_muscle), max(slices_with_muscle)] \
                         if slices_with_muscle else [0, 0]

            # 5.4 存储结果
            results['muscles'][muscle_name] = {
                'zh_name': MUSCLE_NAMES[muscle_name]['zh'],
                'en_name': MUSCLE_NAMES[muscle_name]['en'],
                'volume_mm3': float(volume_mm3),
                'volume_cm3': float(volume_cm3),
                'avg_area_mm2': float(avg_area_mm2),
                'max_area_mm2': float(max_area_mm2),
                'slice_range': slice_range,
                'num_slices': int(len(slices_with_muscle)),
                'slice_areas_mm2': [float(a) for a in slice_areas[c]]
            }

            total_muscle_volume += volume_cm3

        results['total_muscle_volume_cm3'] = float(total_muscle_volume)

        # ========== 第6步: 保存结果 ==========
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._save_results(results, all_predictions, ct_data, ct_nii, output_dir)

        return results, all_predictions
```

**体积计算示意**:
```
一个切片中某肌肉占100个像素
体素尺寸: 1.5 x 1.5 x 1.5 mm

该层面积 = 100 × (1.5 × 1.5) = 225 mm²

如果有50层都有这个肌肉:
总像素数 = 100 × 50 = 5000
体积 = 5000 × (1.5 × 1.5 × 1.5) = 16875 mm³ = 16.875 cm³
```

### 4.5 保存结果 `_save_results`

```python
    def _save_results(self, results, predictions, ct_data, ct_nii, output_dir):
        """
        保存所有分析结果

        输出文件:
        1. muscle_analysis_results.json - JSON格式详细报告
        2. muscle_volumes.csv - CSV表格（Excel友好）
        3. segmentation_mask.nii.gz - 分割掩码（可用3D Slicer查看）
        4. 可视化图片
        """

        # ========== 1. 保存JSON报告 ==========
        json_path = os.path.join(output_dir, 'muscle_analysis_results.json')

        # 创建精简版（不包含每层面积数据，太大）
        results_summary = {k: v for k, v in results.items() if k != 'muscles'}
        results_summary['muscles'] = {}
        for muscle_name, muscle_data in results['muscles'].items():
            # 排除slice_areas_mm2字段
            results_summary['muscles'][muscle_name] = {
                k: v for k, v in muscle_data.items()
                if k != 'slice_areas_mm2'
            }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        # ========== 2. 保存CSV报告 ==========
        csv_path = os.path.join(output_dir, 'muscle_volumes.csv')

        # 构建数据列表
        df_data = []
        for muscle_name, muscle_data in results['muscles'].items():
            df_data.append({
                '肌肉名称(中文)': muscle_data['zh_name'],
                '肌肉名称(英文)': muscle_data['en_name'],
                '体积(cm³)': f"{muscle_data['volume_cm3']:.2f}",
                # ... 其他字段
            })

        # 创建DataFrame并保存
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        # utf-8-sig: 带BOM的UTF-8，Excel能正确识别中文

        # ========== 3. 保存NIfTI分割掩码 ==========
        # 合并10个通道为单个标签图
        # 0=背景, 1=第一种肌肉, 2=第二种肌肉, ...
        combined_mask = np.zeros(predictions.shape[1:], dtype=np.int16)

        for c in range(self.num_classes):
            # 将该类别的像素设为类别ID+1
            combined_mask[predictions[c] > 0.5] = c + 1

        # 创建NIfTI图像，保持原始空间信息
        mask_nii = nib.Nifti1Image(
            combined_mask,
            ct_nii.affine,   # 空间变换矩阵
            ct_nii.header    # 头信息
        )

        nib.save(mask_nii, os.path.join(output_dir, 'segmentation_mask.nii.gz'))

        # ========== 4. 生成可视化 ==========
        self._generate_visualizations(results, predictions, ct_data, output_dir)
```

---

## 5. 主函数流程

```python
def main():
    """
    程序入口点

    执行流程:
    1. 解析命令行参数
    2. 验证输入文件
    3. 创建分析器
    4. 执行分析
    5. 打印报告
    """

    # ========== 第1步: 创建参数解析器 ==========
    parser = argparse.ArgumentParser(description='CT图像肌肉量分析')

    # 必需参数
    parser.add_argument('--ct_path', type=str, required=True,
                        help='CT图像路径 (.nii.gz)')

    # 可选参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--model_path', type=str,
                        default='/local/hzhang02/data/dataset/outputs/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--label_map_path', type=str,
                        default='/local/hzhang02/data/dataset/outputs/label_map.json',
                        help='标签映射文件路径')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (cuda/cpu)')

    # 解析参数
    args = parser.parse_args()

    # ========== 第2步: 验证输入 ==========
    if not os.path.exists(args.ct_path):
        print(f"错误: CT文件不存在: {args.ct_path}")
        return

    # ========== 第3步: 设置默认输出目录 ==========
    if args.output_dir is None:
        ct_dir = os.path.dirname(args.ct_path)
        args.output_dir = os.path.join(ct_dir, 'muscle_analysis')

    # ========== 第4步: 创建分析器 ==========
    analyzer = MuscleAnalyzer(
        model_path=args.model_path,
        label_map_path=args.label_map_path,
        device=args.device
    )

    # ========== 第5步: 执行分析 ==========
    results, predictions = analyzer.analyze_ct(args.ct_path, args.output_dir)

    # ========== 第6步: 打印报告 ==========
    analyzer.print_report(results)

    print(f"\n分析完成！结果已保存到: {args.output_dir}")


# ========== 程序入口 ==========
if __name__ == '__main__':
    # 当直接运行此脚本时（而非被import时）执行main()
    main()
```

**命令行使用示例**:
```bash
# 基本使用
python muscle_analysis.py --ct_path /path/to/ct.nii.gz

# 指定所有参数
python muscle_analysis.py \
    --ct_path /path/to/ct.nii.gz \
    --output_dir /path/to/output \
    --model_path /path/to/model.pth \
    --device cuda
```

---

## 6. 完整执行流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     程序启动                                  │
│                   python muscle_analysis.py                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  解析命令行参数                               │
│         --ct_path, --output_dir, --model_path等              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               创建 MuscleAnalyzer 实例                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 1. 检测设备 (GPU/CPU)                                  │  │
│  │ 2. 加载标签映射 (label_map.json)                       │  │
│  │ 3. 创建U-Net模型                                       │  │
│  │ 4. 加载模型权重 (best_model.pth)                       │  │
│  │ 5. 设置为评估模式                                      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  调用 analyze_ct()                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 1. 加载CT图像 (nibabel)                                │  │
│  │ 2. 获取体素尺寸                                        │  │
│  │ 3. 逐层处理 (循环399次)                                │  │
│  │    ┌─────────────────────────────────────────────┐    │  │
│  │    │ a. 提取切片 ct_data[:,:,z]                   │    │  │
│  │    │ b. preprocess_slice() 预处理                 │    │  │
│  │    │    - 百分位数裁剪                             │    │  │
│  │    │    - 归一化到[0,1]                           │    │  │
│  │    │    - resize到256x256                        │    │  │
│  │    │    - 转为torch tensor                       │    │  │
│  │    │ c. predict_slice() 预测                     │    │  │
│  │    │    - 送入U-Net                              │    │  │
│  │    │    - Sigmoid激活                            │    │  │
│  │    │    - 阈值二值化                             │    │  │
│  │    │ d. resize回原始尺寸                         │    │  │
│  │    │ e. 计算面积                                 │    │  │
│  │    └─────────────────────────────────────────────┘    │  │
│  │ 4. 计算每种肌肉的体积                                  │  │
│  │ 5. 保存结果                                           │  │
│  │    - JSON报告                                         │  │
│  │    - CSV表格                                          │  │
│  │    - NIfTI分割掩码                                    │  │
│  │    - 可视化图片                                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   打印分析报告                               │
│         各肌肉体积、左右对比、统计信息                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      程序结束                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 附录: 关键概念速查

### A. PyTorch张量维度

```
(N, C, H, W)
 │  │  │  └── Width 宽度
 │  │  └───── Height 高度
 │  └──────── Channels 通道数
 └─────────── Batch size 批次大小

例如: (1, 10, 256, 256)
     1张图片，10个通道(10种肌肉)，256x256像素
```

### B. 医学图像体积计算

```
体素(Voxel) = 3D像素
体素尺寸 = (dx, dy, dz) mm

单个体素体积 = dx × dy × dz mm³

肌肉体积 = 肌肉像素数 × 单个体素体积
```

### C. U-Net关键特点

1. **跳跃连接**: 保留低级特征（边缘、纹理）
2. **对称结构**: 编码器提取特征，解码器恢复空间信息
3. **多尺度**: 不同层捕获不同大小的结构

---

*文档完成于 2025-12-17*
