#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
U-Net 2D 医学图像分割模型训练（脂肪组织专用版）

本脚本用于训练基于2D U-Net的脂肪组织分割模型，支持皮下脂肪(SAT)和内脏脂肪(VAT)的分割。

使用两步法：
1. 第一步：基于HU阈值生成脂肪伪标签
2. 第二步：使用伪标签训练U-Net模型进行更精确的分割

数据集说明:
- 数据格式：NIfTI (.nii.gz)
- 输入：CT扫描图像
- 输出：2通道分割掩码 (SAT/VAT)
- 训练方式：2D轴向切片

脂肪类型:
- SAT (Subcutaneous Adipose Tissue): 皮下脂肪，位于皮肤下方到肌肉层之间
- VAT (Visceral Adipose Tissue): 内脏脂肪，位于腹腔内器官周围

HU值范围:
- 脂肪组织: -120 ~ -40 HU
- 空气: < -500 HU
- 软组织: 0 ~ 100 HU

作者: hzhang02
日期: 2024-12-17
"""

# ============================================================================
# 1. 导入依赖库
# ============================================================================

import os
import json
import glob
import random
import time
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage import morphology
from skimage.measure import label, regionprops
from scipy import ndimage

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm

# 实时监控工具
import wandb

import matplotlib.pyplot as plt

# 设置 Weights & Biases API Key（自动登录）
os.environ['WANDB_API_KEY'] = 'e331d01a3e6f2b0b78c22ffde1e676cb4742f891'

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# 2. 脂肪标签定义
# ============================================================================

# HU阈值定义
FAT_HU_MIN = -120  # 脂肪HU最小值
FAT_HU_MAX = -40   # 脂肪HU最大值
AIR_HU_MAX = -500  # 空气HU最大值（排除空气）
BODY_HU_MIN = -500 # 身体区域的最小HU值（排除背景）

# 脂肪类型
FAT_LABELS = [
    'SAT',  # 皮下脂肪 (Subcutaneous Adipose Tissue)
    'VAT',  # 内脏脂肪 (Visceral Adipose Tissue)
]

# 脂肪名称中英法对照
FAT_NAMES = {
    'SAT': {'zh': '皮下脂肪', 'fr': 'Tissu adipeux sous-cutané', 'en': 'Subcutaneous Adipose Tissue'},
    'VAT': {'zh': '内脏脂肪', 'fr': 'Tissu adipeux viscéral', 'en': 'Visceral Adipose Tissue'},
}

print(f"目标脂肪结构数量: {len(FAT_LABELS)}")


# ============================================================================
# 3. 脂肪伪标签生成函数
# ============================================================================

def get_body_mask(slice_img: np.ndarray, threshold: float = -500) -> np.ndarray:
    """
    获取身体区域掩码（排除空气和床板）

    Args:
        slice_img: CT切片 (H, W)，HU值
        threshold: 身体/空气分界阈值

    Returns:
        身体区域掩码 (H, W)
    """
    # 阈值分割
    body = slice_img > threshold

    # 形态学操作清理噪声
    body = morphology.binary_closing(body, morphology.disk(3))
    body = morphology.binary_opening(body, morphology.disk(2))

    # 填充孔洞
    body = ndimage.binary_fill_holes(body)

    # 保留最大连通区域（身体）
    labeled = label(body)
    if labeled.max() > 0:
        regions = regionprops(labeled)
        largest = max(regions, key=lambda x: x.area)
        body = labeled == largest.label

    return body.astype(np.float32)


def get_body_contour(body_mask: np.ndarray, thickness: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取身体外轮廓和腹腔边界

    Args:
        body_mask: 身体区域掩码 (H, W)
        thickness: 轮廓厚度

    Returns:
        outer_contour: 外轮廓掩码
        inner_region: 腹腔内区域掩码
    """
    # 外轮廓 = 身体掩码 - 腐蚀后的身体掩码
    eroded = morphology.binary_erosion(body_mask, morphology.disk(thickness))
    outer_contour = body_mask.astype(bool) & ~eroded

    # 进一步腐蚀得到腹腔内区域（估计）
    inner_region = morphology.binary_erosion(body_mask, morphology.disk(thickness * 3))

    return outer_contour.astype(np.float32), inner_region.astype(np.float32)


def generate_fat_masks(slice_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于HU阈值生成SAT和VAT掩码

    Args:
        slice_img: CT切片 (H, W)，HU值

    Returns:
        sat_mask: 皮下脂肪掩码 (H, W)
        vat_mask: 内脏脂肪掩码 (H, W)
    """
    # 1. 获取身体掩码
    body_mask = get_body_mask(slice_img)

    # 2. 获取脂肪区域（基于HU阈值）
    fat_mask = (slice_img >= FAT_HU_MIN) & (slice_img <= FAT_HU_MAX)
    fat_mask = fat_mask & body_mask.astype(bool)  # 只保留身体内的脂肪

    # 3. 区分SAT和VAT
    # SAT: 位于身体边缘区域的脂肪
    # VAT: 位于身体中心区域的脂肪

    # 获取外围区域（SAT区域）和内部区域（VAT区域）
    outer_contour, inner_region = get_body_contour(body_mask)

    # 外围区域：从皮肤到一定深度
    # 使用多次腐蚀来定义皮下区域
    subcutaneous_depth = 15  # 像素深度，可调整
    outer_region = body_mask.copy()
    eroded = morphology.binary_erosion(body_mask, morphology.disk(subcutaneous_depth))
    outer_region = body_mask.astype(bool) & ~eroded

    # SAT: 外围区域内的脂肪
    sat_mask = fat_mask & outer_region

    # VAT: 内部区域的脂肪（腹腔内）
    vat_mask = fat_mask & eroded

    # 形态学后处理
    sat_mask = morphology.binary_opening(sat_mask, morphology.disk(2))
    sat_mask = morphology.binary_closing(sat_mask, morphology.disk(2))

    vat_mask = morphology.binary_opening(vat_mask, morphology.disk(2))
    vat_mask = morphology.binary_closing(vat_mask, morphology.disk(2))

    return sat_mask.astype(np.float32), vat_mask.astype(np.float32)


def generate_fat_labels_for_volume(ct_data: np.ndarray,
                                   save_path: str = None) -> np.ndarray:
    """
    为整个CT体积生成脂肪伪标签

    Args:
        ct_data: CT体积数据 (H, W, D)，HU值
        save_path: 保存路径（可选）

    Returns:
        labels: 脂肪标签 (2, H, W, D)，通道0=SAT，通道1=VAT
    """
    H, W, D = ct_data.shape
    labels = np.zeros((2, H, W, D), dtype=np.float32)

    for z in range(D):
        slice_img = ct_data[:, :, z]
        sat_mask, vat_mask = generate_fat_masks(slice_img)
        labels[0, :, :, z] = sat_mask
        labels[1, :, :, z] = vat_mask

    if save_path:
        # 保存为NIfTI
        combined = labels[0] + labels[1] * 2  # 1=SAT, 2=VAT
        nii = nib.Nifti1Image(combined.astype(np.int16), np.eye(4))
        nib.save(nii, save_path)

    return labels


print("脂肪伪标签生成函数定义完成")


# ============================================================================
# 4. 数据处理工具函数
# ============================================================================

def find_subjects(root: str) -> List[str]:
    """
    查找所有受试者文件夹（以's'开头）

    Args:
        root: 数据集根目录

    Returns:
        受试者文件夹路径列表
    """
    paths = sorted(glob.glob(os.path.join(root, 's*')))
    return [p for p in paths if os.path.isdir(p)]


print("数据处理函数定义完成")


# ============================================================================
# 5. 数据集类定义
# ============================================================================

class FatSliceDataset(Dataset):
    """
    2D切片数据集（脂肪组织专用版）

    从3D CT体积中提取2D轴向切片，并使用HU阈值方法生成脂肪伪标签
    """

    def __init__(self, subjects: List[str], transform=None,
                 target_shape=(256, 256), cache_labels=True):
        """
        Args:
            subjects: 受试者文件夹路径列表
            transform: 数据增强（未实现）
            target_shape: 目标图像尺寸
            cache_labels: 是否缓存标签
        """
        self.items = []
        self.transform = transform
        self.target_shape = target_shape
        self.cache_labels = cache_labels
        self.label_cache = {}

        print("正在构建脂肪数据集...")
        for s in tqdm(subjects, desc="加载受试者"):
            ct_path = os.path.join(s, 'ct.nii.gz')
            if not os.path.exists(ct_path):
                continue

            # 加载CT获取深度信息
            img = nib.load(ct_path)
            data = img.get_fdata().astype(np.float32)
            depth = data.shape[2]

            # 为每个切片创建一个样本
            for z in range(depth):
                self.items.append((ct_path, z))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ct_path, z = self.items[idx]

        # 检查缓存
        cache_key = f"{ct_path}_{z}"
        if self.cache_labels and cache_key in self.label_cache:
            return self.label_cache[cache_key]

        # 加载CT切片
        img = nib.load(ct_path).get_fdata().astype(np.float32)
        slice_img = img[:, :, z]

        # 生成脂肪伪标签（使用原始HU值）
        sat_mask, vat_mask = generate_fat_masks(slice_img)

        # 归一化CT图像（用于模型输入）
        lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
        slice_normalized = np.clip(slice_img, lo, hi)
        if hi - lo > 0:
            slice_normalized = (slice_normalized - lo) / (hi - lo)
        else:
            slice_normalized = np.zeros_like(slice_normalized)

        # 调整大小
        H, W = self.target_shape
        slice_normalized = resize(slice_normalized, (H, W), order=1,
                                  preserve_range=True, anti_aliasing=True)
        sat_mask = resize(sat_mask, (H, W), order=0,
                         preserve_range=True, anti_aliasing=False)
        vat_mask = resize(vat_mask, (H, W), order=0,
                         preserve_range=True, anti_aliasing=False)

        # 二值化掩码
        sat_mask = (sat_mask > 0.5).astype(np.float32)
        vat_mask = (vat_mask > 0.5).astype(np.float32)

        # 构建多通道掩码 (2, H, W)
        mask = np.stack([sat_mask, vat_mask], axis=0)

        # 转换为tensor
        img_t = torch.from_numpy(slice_normalized).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).float()

        # 缓存结果
        if self.cache_labels:
            self.label_cache[cache_key] = (img_t, mask_t)

        return img_t, mask_t


print("数据集类定义完成")


# ============================================================================
# 6. U-Net模型定义
# ============================================================================

class DoubleConv(nn.Module):
    """双卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    """2D U-Net模型"""

    def __init__(self, in_ch=1, out_ch=2, features=[32, 64, 128, 256]):
        """
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数（分割类别数，默认2: SAT/VAT）
            features: 每一层的特征数
        """
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 编码器
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器
        rev = list(reversed(features))
        up_in = features[-1] * 2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(up_in, f))
            up_in = f

        # 最终卷积层
        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        # 编码路径
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码路径
        for idx in range(0, len(self.ups), 2):
            trans = self.ups[idx]
            conv = self.ups[idx + 1]
            x = trans(x)
            skip = skips[-(idx // 2) - 1]
            if x.shape != skip.shape:
                # 中心裁剪skip以匹配x
                _, _, h, w = x.shape
                skip = skip[:, :, :h, :w]
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.final(x)


print("U-Net模型定义完成")


# ============================================================================
# 7. 评估指标
# ============================================================================

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    计算Dice系数

    Args:
        pred: 预测掩码 (N, C, H, W)
        target: 真实掩码 (N, C, H, W)
        eps: 平滑项

    Returns:
        平均Dice系数
    """
    N, C = pred.shape[:2]
    pred = pred.view(N, C, -1)
    target = target.view(N, C, -1)
    inter = (pred * target).sum(-1)
    union = pred.sum(-1) + target.sum(-1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def dice_score_per_class(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    计算每个类别的Dice系数

    Args:
        pred: 预测掩码 (N, C, H, W)
        target: 真实掩码 (N, C, H, W)
        eps: 平滑项

    Returns:
        每个类别的Dice系数 (C,)
    """
    N, C = pred.shape[:2]
    pred = pred.view(N, C, -1)
    target = target.view(N, C, -1)
    inter = (pred * target).sum(dim=(0, 2))
    union = pred.sum(dim=(0, 2)) + target.sum(dim=(0, 2))
    dice = (2 * inter + eps) / (union + eps)
    return dice


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    计算IoU (Intersection over Union)

    Args:
        pred: 预测掩码 (N, C, H, W)
        target: 真实掩码 (N, C, H, W)
        eps: 平滑项

    Returns:
        平均IoU分数
    """
    N, C = pred.shape[:2]
    pred = pred.view(N, C, -1)
    target = target.view(N, C, -1)
    inter = (pred * target).sum(-1)
    union = pred.sum(-1) + target.sum(-1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def compute_gradient_norm(model):
    """
    计算模型梯度的L2范数
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


print("评估指标函数定义完成")


# ============================================================================
# 8. 数据准备
# ============================================================================

# 基础配置
DATA_ROOT = '/local/hzhang02/data'
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs_fat'
TARGET_SHAPE = (256, 256)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10  # 减少epoch数加快训练
NUM_SUBJECTS = 50  # 最多使用50个受试者

# 早停和阈值配置
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 3  # 减少patience加快停止
EARLY_STOP_MIN_DELTA = 0.001

# 准确率阈值停止配置
USE_ACCURACY_THRESHOLD = True
ACCURACY_THRESHOLD = 0.80  # 降低阈值加快停止
ACCURACY_THRESHOLD_PATIENCE = 2

# 学习率调度器配置
USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'
COSINE_T_MAX = 10  # 与EPOCHS匹配
COSINE_ETA_MIN = 1e-6

# 实时监控配置
USE_WANDB = True
WANDB_PROJECT = 'fat-segmentation-unet'
WANDB_RUN_NAME = 'unet-2d-fat-training'
WANDB_NOTES = '2D U-Net training for fat tissue segmentation (SAT/VAT)'

# 日志记录频率
LOG_EVERY_N_BATCHES = 10
LOG_IMAGES_EVERY_N_EPOCHS = 1

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("配置信息（脂肪分割专用版）:")
print(f"  数据根目录: {DATA_ROOT}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  目标结构: 2种脂肪组织 (SAT/VAT)")
print(f"  批次大小: {BATCH_SIZE}")
print(f"  学习率: {LEARNING_RATE}")
print(f"  最大训练轮数: {EPOCHS}")
print(f"  早停机制: {'启用 (Patience=' + str(EARLY_STOP_PATIENCE) + ')' if USE_EARLY_STOPPING else '禁用'}")
print(f"  准确率阈值停止: {'启用 (Threshold=' + str(ACCURACY_THRESHOLD) + ')' if USE_ACCURACY_THRESHOLD else '禁用'}")
print(f"  实时监控: {'启用 (Weights & Biases)' if USE_WANDB else '禁用'}")
print("=" * 60)

# 查找受试者，只使用前 NUM_SUBJECTS 个
all_subjects_raw = [d for d in os.listdir(DATA_ROOT) if d.startswith('s') and os.path.isdir(os.path.join(DATA_ROOT, d))]
all_subjects = [s for s in all_subjects_raw if s in [f's{i:04d}' for i in range(NUM_SUBJECTS)]]
subjects = [os.path.join(DATA_ROOT, s) for s in sorted(all_subjects)]

print(f"\n找到 {len(subjects)} 个受试者")

# 标签映射
label_map = {
    'SAT': 0,  # 皮下脂肪
    'VAT': 1,  # 内脏脂肪
}
idx_to_label = {v: k for k, v in label_map.items()}

print(f"脂肪结构数量: {len(label_map)}")
print("目标脂肪列表:")
for name, idx in label_map.items():
    print(f"  [{idx}] {FAT_NAMES[name]['zh']} / {FAT_NAMES[name]['en']}")

# 保存标签映射
label_map_path = os.path.join(OUTPUT_DIR, 'fat_label_map.json')
with open(label_map_path, 'w') as f:
    json.dump(label_map, f, indent=2)
print(f"标签映射已保存到: {label_map_path}")

# 划分训练集和验证集
random.shuffle(subjects)
n = len(subjects)
ntrain = max(1, int(n * 0.8))
train_subs = subjects[:ntrain]
val_subs = subjects[ntrain:]

print(f"\n训练集受试者数量: {len(train_subs)}")
print(f"验证集受试者数量: {len(val_subs)}")
print(f"数据分割比例: 训练{len(train_subs)/n:.1%} / 验证{len(val_subs)/n:.1%}")

# 创建数据集
print("\n创建训练数据集（使用HU阈值生成伪标签）...")
train_ds = FatSliceDataset(train_subs, target_shape=TARGET_SHAPE, cache_labels=False)
print(f"训练切片数量: {len(train_ds)}")

print("\n创建验证数据集...")
val_ds = FatSliceDataset(val_subs, target_shape=TARGET_SHAPE, cache_labels=False)
print(f"验证切片数量: {len(val_ds)}")

# 创建数据加载器
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

print(f"\n训练批次数量: {len(train_loader)}")
print(f"验证批次数量: {len(val_loader)}")


# ============================================================================
# 9. 可视化样本
# ============================================================================

# 获取一个样本
sample_img, sample_mask = train_ds[len(train_ds) // 2]

print(f"\n图像形状: {sample_img.shape}")
print(f"掩码形状: {sample_mask.shape}")

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 原始CT图像
axes[0].imshow(sample_img[0], cmap='gray')
axes[0].set_title('CT Image')
axes[0].axis('off')

# SAT掩码
axes[1].imshow(sample_img[0], cmap='gray')
axes[1].imshow(sample_mask[0], alpha=0.5, cmap='Reds')
axes[1].set_title('SAT (皮下脂肪)')
axes[1].axis('off')

# VAT掩码
axes[2].imshow(sample_img[0], cmap='gray')
axes[2].imshow(sample_mask[1], alpha=0.5, cmap='Blues')
axes[2].set_title('VAT (内脏脂肪)')
axes[2].axis('off')

# 两种脂肪叠加
axes[3].imshow(sample_img[0], cmap='gray')
axes[3].imshow(sample_mask[0], alpha=0.4, cmap='Reds')
axes[3].imshow(sample_mask[1], alpha=0.4, cmap='Blues')
axes[3].set_title('SAT (红) + VAT (蓝)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_visualization_fat.png'), dpi=100, bbox_inches='tight')
plt.close()
print("\n样本可视化已保存")


# ============================================================================
# 10. 创建模型
# ============================================================================

# 创建模型（2个输出通道：SAT和VAT）
model = UNet2D(in_ch=1, out_ch=2, features=[32, 64, 128, 256]).to(device)

# 优化器和损失函数
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# 学习率调度器
if USE_SCHEDULER:
    if SCHEDULER_TYPE == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=COSINE_T_MAX,
            eta_min=COSINE_ETA_MIN
        )
        print(f"\n使用余弦退火学习率调度器")
        print(f"  学习率范围: {LEARNING_RATE} -> {COSINE_ETA_MIN}")
        print(f"  周期: {COSINE_T_MAX} epochs")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=True
        )
        print(f"\n使用ReduceLROnPlateau学习率调度器")
else:
    scheduler = None
    print(f"\n使用固定学习率: {LEARNING_RATE}")

# 打印模型信息
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n模型参数统计:")
print(f"  总参数量: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")
print(f"\n模型已创建并移至 {device}")


# ============================================================================
# 11. 初始化 Weights & Biases
# ============================================================================

if USE_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        notes=WANDB_NOTES,
        config={
            'architecture': 'U-Net 2D',
            'dataset': 'Medical CT Fat Segmentation',
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss': 'BCEWithLogitsLoss',
            'use_scheduler': USE_SCHEDULER,
            'scheduler_type': SCHEDULER_TYPE if USE_SCHEDULER else 'none',
            'input_shape': TARGET_SHAPE,
            'num_classes': len(label_map),
            'fat_hu_range': f'{FAT_HU_MIN} to {FAT_HU_MAX}',
            'train_subjects': len(train_subs),
            'val_subjects': len(val_subs),
            'train_slices': len(train_ds),
            'val_slices': len(val_ds),
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
    )
    wandb.watch(model, log='all', log_freq=100)
    print("\nWeights & Biases 已初始化")
    print(f"项目: {WANDB_PROJECT}")
    print(f"运行名称: {WANDB_RUN_NAME}")


# ============================================================================
# 12. 训练模型
# ============================================================================

# 训练历史记录
history = {
    'train_loss': [],
    'train_dice': [],
    'val_loss': [],
    'val_dice': [],
    'val_iou': [],
    'sat_dice': [],
    'vat_dice': [],
    'learning_rate': [],
    'gradient_norm': [],
    'epoch_time': [],
}

# 早停相关变量
best_val_dice = 0.0
best_epoch = 0
epochs_no_improve = 0
epochs_above_threshold = 0
early_stop_triggered = False
threshold_stop_triggered = False

print(f"\n开始训练（最多 {EPOCHS} 个epoch）...\n")

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()

    print(f"{'=' * 60}")
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"{'=' * 60}")

    # ========== 训练阶段 ==========
    model.train()
    running_loss = 0.0
    train_dice_scores = []
    train_bar = tqdm(train_loader, desc='Training')

    for batch_idx, (imgs, masks) in enumerate(train_bar):
        imgs = imgs.to(device)
        masks = masks.to(device)

        # 前向传播
        preds = model(imgs)
        loss = criterion(preds, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数
        grad_norm = compute_gradient_norm(model)

        optimizer.step()

        # 计算训练Dice
        with torch.no_grad():
            preds_sigmoid = torch.sigmoid(preds)
            preds_bin = (preds_sigmoid > 0.5).float()
            train_dice = dice_score(preds_bin, masks)
            train_dice_scores.append(train_dice)

        running_loss += loss.item()
        train_bar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{train_dice:.4f}'})

        # 实时记录训练指标
        if USE_WANDB and (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0:
            wandb.log({
                'batch/loss': loss.item(),
                'batch/dice': train_dice,
                'batch/gradient_norm': grad_norm,
                'epoch': epoch,
                'batch': batch_idx + 1,
            })

    avg_train_loss = running_loss / len(train_loader)
    avg_train_dice = float(np.mean(train_dice_scores))
    history['train_loss'].append(avg_train_loss)
    history['train_dice'].append(avg_train_dice)
    history['gradient_norm'].append(grad_norm)

    # ========== 验证阶段 ==========
    model.eval()
    val_running_loss = 0.0
    val_dice_scores = []
    val_iou_scores = []
    sat_dice_scores = []
    vat_dice_scores = []
    val_bar = tqdm(val_loader, desc='Validation')

    sample_images = []
    sample_masks = []
    sample_preds = []

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(val_bar):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # 前向传播
            preds = model(imgs)
            val_loss = criterion(preds, masks)
            val_running_loss += val_loss.item()

            # 计算指标
            preds_sigmoid = torch.sigmoid(preds)
            preds_bin = (preds_sigmoid > 0.5).float()

            dice = dice_score(preds_bin, masks)
            iou = iou_score(preds_bin, masks)

            val_dice_scores.append(dice)
            val_iou_scores.append(iou)

            # 计算每个类别的dice
            class_dices = dice_score_per_class(preds_bin, masks)
            sat_dice_scores.append(class_dices[0].item())
            vat_dice_scores.append(class_dices[1].item())

            val_bar.set_postfix({'dice': f'{dice:.4f}', 'iou': f'{iou:.4f}'})

            # 保存第一个batch的样本用于可视化
            if batch_idx == 0 and USE_WANDB and epoch % LOG_IMAGES_EVERY_N_EPOCHS == 0:
                sample_images = imgs.cpu()
                sample_masks = masks.cpu()
                sample_preds = preds_bin.cpu()

    avg_val_loss = val_running_loss / len(val_loader)
    avg_val_dice = float(np.mean(val_dice_scores))
    avg_val_iou = float(np.mean(val_iou_scores))
    avg_sat_dice = float(np.mean(sat_dice_scores))
    avg_vat_dice = float(np.mean(vat_dice_scores))

    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)
    history['val_iou'].append(avg_val_iou)
    history['sat_dice'].append(avg_sat_dice)
    history['vat_dice'].append(avg_vat_dice)

    # 计算epoch时间
    epoch_time = time.time() - epoch_start_time
    history['epoch_time'].append(epoch_time)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])

    # 过拟合分析
    overfit_gap = avg_train_dice - avg_val_dice

    # ========== 打印结果 ==========
    print(f"\nEpoch {epoch} 结果:")
    print(f"  训练损失: {avg_train_loss:.4f} | 训练Dice: {avg_train_dice:.4f}")
    print(f"  验证损失: {avg_val_loss:.4f} | 验证Dice: {avg_val_dice:.4f} | 验证IoU: {avg_val_iou:.4f}")
    print(f"  SAT Dice: {avg_sat_dice:.4f} | VAT Dice: {avg_vat_dice:.4f}")
    print(f"  过拟合差距: {overfit_gap:.4f} ({'过拟合' if overfit_gap > 0.1 else '正常'})")
    print(f"  Epoch耗时: {epoch_time:.1f}秒")

    # ========== 记录到 wandb ==========
    if USE_WANDB:
        log_dict = {
            'epoch': epoch,
            'train/loss': avg_train_loss,
            'train/dice': avg_train_dice,
            'val/loss': avg_val_loss,
            'val/dice': avg_val_dice,
            'val/iou': avg_val_iou,
            'fat/sat_dice': avg_sat_dice,
            'fat/vat_dice': avg_vat_dice,
            'comparison/overfit_score': overfit_gap,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_norm': grad_norm,
            'epoch_time': epoch_time,
        }

        # 记录样本图像
        if epoch % LOG_IMAGES_EVERY_N_EPOCHS == 0 and len(sample_images) > 0:
            img_sample = sample_images[0, 0].numpy()
            sat_gt = sample_masks[0, 0].numpy()
            vat_gt = sample_masks[0, 1].numpy()
            sat_pred = sample_preds[0, 0].numpy()
            vat_pred = sample_preds[0, 1].numpy()

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            axes[0, 0].imshow(img_sample, cmap='gray')
            axes[0, 0].set_title('CT Image')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(img_sample, cmap='gray')
            axes[0, 1].imshow(sat_gt, alpha=0.5, cmap='Reds')
            axes[0, 1].set_title('SAT Ground Truth')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(img_sample, cmap='gray')
            axes[0, 2].imshow(sat_pred, alpha=0.5, cmap='Reds')
            axes[0, 2].set_title(f'SAT Prediction (Dice: {avg_sat_dice:.3f})')
            axes[0, 2].axis('off')

            axes[1, 0].imshow(img_sample, cmap='gray')
            axes[1, 0].set_title('CT Image')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(img_sample, cmap='gray')
            axes[1, 1].imshow(vat_gt, alpha=0.5, cmap='Blues')
            axes[1, 1].set_title('VAT Ground Truth')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(img_sample, cmap='gray')
            axes[1, 2].imshow(vat_pred, alpha=0.5, cmap='Blues')
            axes[1, 2].set_title(f'VAT Prediction (Dice: {avg_vat_dice:.3f})')
            axes[1, 2].axis('off')

            plt.tight_layout()
            log_dict['predictions'] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_dict)

    # ========== 保存检查点 ==========
    checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_fat_epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'train_dice': avg_train_dice,
        'val_loss': avg_val_loss,
        'val_dice': avg_val_dice,
        'val_iou': avg_val_iou,
        'sat_dice': avg_sat_dice,
        'vat_dice': avg_vat_dice,
    }, checkpoint_path)
    print(f"  检查点已保存: {checkpoint_path}")

    # 记录最佳模型
    if USE_WANDB and avg_val_dice == max(history['val_dice']):
        wandb.run.summary["best_val_dice"] = avg_val_dice
        wandb.run.summary["best_val_iou"] = avg_val_iou
        wandb.run.summary["best_epoch"] = epoch

    # ========== 更新学习率 ==========
    if USE_SCHEDULER and scheduler is not None:
        if SCHEDULER_TYPE == 'cosine':
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  学习率已更新: {current_lr:.8f}")
        else:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_dice)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  学习率已降低: {old_lr:.8f} -> {new_lr:.8f}")

    # ========== 早停检查 ==========
    if avg_val_dice > best_val_dice + EARLY_STOP_MIN_DELTA:
        best_val_dice = avg_val_dice
        best_epoch = epoch
        epochs_no_improve = 0
        print(f"  验证Dice提升！新的最佳: {best_val_dice:.4f}")

        # 保存最佳模型
        best_model_path = os.path.join(OUTPUT_DIR, 'best_fat_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': avg_val_dice,
            'val_iou': avg_val_iou,
            'sat_dice': avg_sat_dice,
            'vat_dice': avg_vat_dice,
        }, best_model_path)
    else:
        epochs_no_improve += 1
        print(f"  验证Dice无改善（已{epochs_no_improve}/{EARLY_STOP_PATIENCE}轮）")

    # 检查准确率阈值
    if USE_ACCURACY_THRESHOLD and avg_val_dice >= ACCURACY_THRESHOLD:
        epochs_above_threshold += 1
        print(f"  已达到准确率阈值 {ACCURACY_THRESHOLD:.4f}！({epochs_above_threshold}/{ACCURACY_THRESHOLD_PATIENCE}轮)")

        if epochs_above_threshold >= ACCURACY_THRESHOLD_PATIENCE:
            threshold_stop_triggered = True
            print(f"\n{'=' * 60}")
            print(f"准确率阈值停止触发！")
            print(f"  验证Dice已达到 {avg_val_dice:.4f} >= {ACCURACY_THRESHOLD:.4f}")
            print(f"{'=' * 60}\n")
    else:
        epochs_above_threshold = 0

    # 检查早停条件
    if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOP_PATIENCE:
        early_stop_triggered = True
        print(f"\n{'=' * 60}")
        print(f"早停触发！")
        print(f"  验证Dice已连续 {EARLY_STOP_PATIENCE} 个epoch无改善")
        print(f"  最佳验证Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
        print(f"{'=' * 60}\n")

    print()

    # 跳出训练循环
    if early_stop_triggered or threshold_stop_triggered:
        break

print(f"\n{'=' * 60}")
print("训练完成！")
if early_stop_triggered:
    print(f"停止原因: 早停机制")
elif threshold_stop_triggered:
    print(f"停止原因: 达到准确率阈值")
else:
    print(f"停止原因: 完成所有训练轮数")
print(f"实际训练轮数: {len(history['train_loss'])}/{EPOCHS}")
if USE_WANDB:
    print(f"查看完整训练报告: {wandb.run.get_url()}")
print(f"{'=' * 60}")


# ============================================================================
# 13. 训练历史可视化
# ============================================================================

actual_epochs = len(history['train_loss'])

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 训练损失 vs 验证损失
axes[0, 0].plot(range(1, actual_epochs + 1), history['train_loss'], marker='o', label='Train Loss', linewidth=2)
axes[0, 0].plot(range(1, actual_epochs + 1), history['val_loss'], marker='s', label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. 训练Dice vs 验证Dice
axes[0, 1].plot(range(1, actual_epochs + 1), history['train_dice'], marker='o', label='Train Dice', linewidth=2, color='green')
axes[0, 1].plot(range(1, actual_epochs + 1), history['val_dice'], marker='s', label='Val Dice', linewidth=2, color='orange')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Dice Score')
axes[0, 1].set_title('Dice Score Comparison')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. SAT vs VAT Dice
axes[0, 2].plot(range(1, actual_epochs + 1), history['sat_dice'], marker='o', label='SAT Dice', linewidth=2, color='red')
axes[0, 2].plot(range(1, actual_epochs + 1), history['vat_dice'], marker='s', label='VAT Dice', linewidth=2, color='blue')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Dice Score')
axes[0, 2].set_title('SAT vs VAT Dice')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# 4. 验证IoU
axes[1, 0].plot(range(1, actual_epochs + 1), history['val_iou'], marker='d', linewidth=2, color='purple')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('IoU Score')
axes[1, 0].set_title('Validation IoU')
axes[1, 0].grid(alpha=0.3)

# 5. 梯度范数
axes[1, 1].plot(range(1, actual_epochs + 1), history['gradient_norm'], marker='o', linewidth=2, color='brown')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Gradient Norm')
axes[1, 1].set_title('Gradient Norm')
axes[1, 1].grid(alpha=0.3)

# 6. 学习率
axes[1, 2].plot(range(1, actual_epochs + 1), history['learning_rate'], marker='o', linewidth=2, color='blue')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Learning Rate')
axes[1, 2].set_title('Learning Rate Schedule')
axes[1, 2].set_yscale('log')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_fat.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n训练历史图已保存")


# ============================================================================
# 14. 保存训练历史
# ============================================================================

history_path = os.path.join(OUTPUT_DIR, 'training_history_fat.json')
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print(f"训练历史已保存到: {history_path}")

# 打印最终结果摘要
print("\n" + "=" * 60)
print("训练摘要:")
print(f"  最终训练损失: {history['train_loss'][-1]:.4f}")
print(f"  最终训练Dice: {history['train_dice'][-1]:.4f}")
print(f"  最终验证损失: {history['val_loss'][-1]:.4f}")
print(f"  最终验证Dice: {history['val_dice'][-1]:.4f}")
print(f"  最终验证IoU: {history['val_iou'][-1]:.4f}")
print(f"  最终SAT Dice: {history['sat_dice'][-1]:.4f}")
print(f"  最终VAT Dice: {history['vat_dice'][-1]:.4f}")
print(f"  最佳验证Dice: {max(history['val_dice']):.4f} (Epoch {history['val_dice'].index(max(history['val_dice'])) + 1})")
print(f"  总训练时间: {sum(history['epoch_time']):.1f}秒 ({sum(history['epoch_time'])/60:.1f}分钟)")
print("=" * 60)


# ============================================================================
# 15. 测试推理
# ============================================================================

model.eval()
test_img, test_mask = val_ds[len(val_ds) // 2]
test_img_batch = test_img.unsqueeze(0).to(device)

with torch.no_grad():
    test_pred = torch.sigmoid(model(test_img_batch))
    test_pred_bin = (test_pred > 0.5).float()

test_img = test_img.cpu()
test_mask = test_mask.cpu()
test_pred_bin = test_pred_bin[0].cpu()

# 可视化预测结果
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 第一行：SAT
axes[0, 0].imshow(test_img[0], cmap='gray')
axes[0, 0].set_title('CT Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(test_img[0], cmap='gray')
axes[0, 1].imshow(test_mask[0], alpha=0.5, cmap='Reds')
axes[0, 1].set_title('SAT Ground Truth')
axes[0, 1].axis('off')

axes[0, 2].imshow(test_img[0], cmap='gray')
axes[0, 2].imshow(test_pred_bin[0], alpha=0.5, cmap='Reds')
axes[0, 2].set_title('SAT Prediction')
axes[0, 2].axis('off')

# SAT对比
axes[0, 3].imshow(test_img[0], cmap='gray')
axes[0, 3].contour(test_mask[0], colors='red', linewidths=2, alpha=0.7)
axes[0, 3].contour(test_pred_bin[0], colors='green', linewidths=2, alpha=0.7, linestyles='dashed')
axes[0, 3].set_title('SAT: GT (Red) vs Pred (Green)')
axes[0, 3].axis('off')

# 第二行：VAT
axes[1, 0].imshow(test_img[0], cmap='gray')
axes[1, 0].set_title('CT Image')
axes[1, 0].axis('off')

axes[1, 1].imshow(test_img[0], cmap='gray')
axes[1, 1].imshow(test_mask[1], alpha=0.5, cmap='Blues')
axes[1, 1].set_title('VAT Ground Truth')
axes[1, 1].axis('off')

axes[1, 2].imshow(test_img[0], cmap='gray')
axes[1, 2].imshow(test_pred_bin[1], alpha=0.5, cmap='Blues')
axes[1, 2].set_title('VAT Prediction')
axes[1, 2].axis('off')

# VAT对比
axes[1, 3].imshow(test_img[0], cmap='gray')
axes[1, 3].contour(test_mask[1], colors='blue', linewidths=2, alpha=0.7)
axes[1, 3].contour(test_pred_bin[1], colors='cyan', linewidths=2, alpha=0.7, linestyles='dashed')
axes[1, 3].set_title('VAT: GT (Blue) vs Pred (Cyan)')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'test_inference_fat.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n测试推理结果已保存")

# 计算测试样本的Dice
sample_dice = dice_score(test_pred_bin.unsqueeze(0), test_mask.unsqueeze(0))
print(f"测试样本Dice系数: {sample_dice:.4f}")


# ============================================================================
# 16. 关闭实时监控
# ============================================================================

if USE_WANDB:
    wandb.finish()
    print("\nWeights & Biases 会话已关闭")


# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 60)
print("脂肪分割专用版训练脚本执行完成")
print("\n本脚本的主要特性：")
print("【目标结构】")
print("  2种脂肪组织：")
print("  - SAT (Subcutaneous Adipose Tissue): 皮下脂肪")
print("  - VAT (Visceral Adipose Tissue): 内脏脂肪")
print("\n【伪标签生成】")
print(f"  脂肪HU范围: {FAT_HU_MIN} ~ {FAT_HU_MAX} HU")
print("  使用形态学操作区分SAT和VAT")
print("\n【监控指标】")
print("1. 训练/验证Dice - 整体分割效果")
print("2. SAT Dice - 皮下脂肪分割效果")
print("3. VAT Dice - 内脏脂肪分割效果")
print("4. IoU分数 - 另一个常用的分割指标")
print("5. 梯度范数 - 训练稳定性")
print("=" * 60)
