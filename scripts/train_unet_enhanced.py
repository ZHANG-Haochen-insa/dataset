#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
U-Net 2D 医学图像分割模型训练（增强版 - 更多监控指标）

本脚本用于训练基于2D U-Net的医学图像分割模型，并支持使用Weights & Biases进行实时监控。
增强版添加了更多有用的训练监控指标。

数据集说明:
- 数据格式：NIfTI (.nii.gz)
- 输入：CT扫描图像
- 输出：117个解剖结构的多通道分割掩码
- 训练方式：2D轴向切片

实时监控:
本脚本集成了Weights & Biases (wandb)实时监控功能，可以在云端查看训练进度。
"""

# ============================================================================
# 1. 导入依赖库
# ============================================================================

import os
import json
import glob
import random
import time
from typing import List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import nibabel as nib
from skimage.transform import resize

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
# 2. 数据处理工具函数
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


def build_label_map(subjects: List[str], seg_subfolder='segmentations'):
    """
    构建标签映射（文件名 -> 通道索引）

    Args:
        subjects: 受试者文件夹列表
        seg_subfolder: 分割文件子文件夹名称

    Returns:
        dict: 标签名到通道索引的映射
    """
    names = set()
    for s in subjects:
        segdir = os.path.join(s, seg_subfolder)
        if os.path.isdir(segdir):
            for p in glob.glob(os.path.join(segdir, '*.nii*')):
                names.add(os.path.basename(p))
    names = sorted(names)
    label_map = {name: idx for idx, name in enumerate(names)}
    return label_map


print("数据处理函数定义完成")


# ============================================================================
# 3. 数据集类定义
# ============================================================================

class SliceDataset(Dataset):
    """
    2D切片数据集

    从3D CT体积中提取2D轴向切片，并加载对应的多通道分割掩码
    """

    def __init__(self, subjects: List[str], label_map: dict,
                 seg_subfolder='segmentations', transform=None, target_shape=(256,256)):
        """
        Args:
            subjects: 受试者文件夹路径列表
            label_map: 标签映射字典
            seg_subfolder: 分割文件夹名称
            transform: 数据增强（未实现）
            target_shape: 目标图像尺寸
        """
        self.items = []
        self.label_map = label_map
        self.transform = transform
        self.target_shape = target_shape

        for s in subjects:
            ct_path = os.path.join(s, 'ct.nii.gz')
            segdir = os.path.join(s, seg_subfolder)
            if not os.path.exists(ct_path):
                continue

            # 查找该受试者的分割文件
            segfiles = {}
            if os.path.isdir(segdir):
                for p in glob.glob(os.path.join(segdir, '*.nii*')):
                    name = os.path.basename(p)
                    if name in label_map:
                        segfiles[label_map[name]] = p

            # 获取切片数量
            img = nib.load(ct_path)
            data = img.get_fdata().astype(np.float32)
            depth = data.shape[2]

            # 为每个切片创建一个样本
            for z in range(depth):
                self.items.append((ct_path, segfiles, z))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ct_path, segfiles, z = self.items[idx]

        # 加载CT切片
        img = nib.load(ct_path).get_fdata().astype(np.float32)
        slice_img = img[:, :, z]

        # 使用百分位数进行窗口化归一化
        lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
        slice_img = np.clip(slice_img, lo, hi)
        if hi - lo > 0:
            slice_img = (slice_img - lo) / (hi - lo)
        else:
            slice_img = np.zeros_like(slice_img)

        # 调整大小
        H, W = self.target_shape
        slice_img = resize(slice_img, (H, W), order=1, preserve_range=True, anti_aliasing=True)

        # 构建多通道掩码
        C = len(self.label_map)
        mask = np.zeros((C, H, W), dtype=np.float32)
        for ch, p in segfiles.items():
            m = nib.load(p).get_fdata().astype(np.float32)
            m_slice = m[:, :, z]
            m_slice = resize(m_slice, (H, W), order=0, preserve_range=True, anti_aliasing=False)
            mask[ch] = (m_slice > 0.5).astype(np.float32)

        # 转换为tensor
        img_t = torch.from_numpy(slice_img).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).float()
        return img_t, mask_t


print("数据集类定义完成")


# ============================================================================
# 4. U-Net模型定义
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

    def __init__(self, in_ch=1, out_ch=1, features=[32,64,128,256]):
        """
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数（分割类别数）
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
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # 解码器
        rev = list(reversed(features))
        up_in = features[-1]*2
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
            conv = self.ups[idx+1]
            x = trans(x)
            skip = skips[-(idx//2)-1]
            if x.shape != skip.shape:
                # 中心裁剪skip以匹配x
                _,_,h,w = x.shape
                skip = skip[:, :, :h, :w]
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.final(x)


print("U-Net模型定义完成")


# ============================================================================
# 5. 评估指标（增强版）
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

    Args:
        model: PyTorch模型

    Returns:
        梯度范数
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
# 6. 数据准备
# ============================================================================

# 基础配置
DATA_ROOT = '/local/hzhang02/data'
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs'
TARGET_SHAPE = (256, 256)
BATCH_SIZE = 16  # 增加到16利用24GB显存（原来是8）
LEARNING_RATE = 1e-3
EPOCHS = 20  # 增加训练轮数以获得更好效果

# 学习率调度器配置
USE_SCHEDULER = True  # 是否使用学习率调度器
SCHEDULER_TYPE = 'cosine'  # 'cosine' 或 'plateau'
# Cosine参数
COSINE_T_MAX = 20  # 余弦周期（通常等于总epoch数）
COSINE_ETA_MIN = 1e-6  # 最小学习率
# Plateau参数
PLATEAU_FACTOR = 0.5  # 学习率衰减因子
PLATEAU_PATIENCE = 3  # 容忍多少个epoch性能不提升
PLATEAU_MIN_LR = 1e-6  # 最小学习率

# 实时监控配置
USE_WANDB = True
WANDB_PROJECT = 'medical-segmentation-unet'
WANDB_RUN_NAME = 'unet-2d-training-enhanced'
WANDB_NOTES = '2D U-Net training with enhanced monitoring metrics'

# 日志记录频率
LOG_EVERY_N_BATCHES = 10
LOG_IMAGES_EVERY_N_EPOCHS = 1

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("配置信息:")
print(f"  数据根目录: {DATA_ROOT}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  批次大小: {BATCH_SIZE}")
print(f"  学习率: {LEARNING_RATE}")
print(f"  训练轮数: {EPOCHS}")
print(f"  实时监控: {'启用 (Weights & Biases - 增强版)' if USE_WANDB else '禁用'}")
print("=" * 60)

# 查找受试者，只使用 s0000 到 s0100
all_subjects_raw = [d for d in os.listdir(DATA_ROOT) if d.startswith('s') and os.path.isdir(os.path.join(DATA_ROOT, d))]
all_subjects = [s for s in all_subjects_raw if s in [f's{i:04d}' for i in range(101)]]
subjects = [os.path.join(DATA_ROOT, s) for s in sorted(all_subjects)]

print(f"\n找到 {len(subjects)} 个受试者")

# 构建标签映射
label_map = build_label_map(subjects)
print(f"解剖结构数量: {len(label_map)}")

# 保存标签映射
label_map_path = os.path.join(OUTPUT_DIR, 'label_map.json')
with open(label_map_path, 'w') as f:
    json.dump(label_map, f, indent=2)
print(f"标签映射已保存到: {label_map_path}")

# 创建反向映射（索引 -> 名称）
idx_to_label = {v: k.replace('.nii.gz', '') for k, v in label_map.items()}

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
print("\n创建训练数据集...")
train_ds = SliceDataset(train_subs, label_map, target_shape=TARGET_SHAPE)
print(f"训练切片数量: {len(train_ds)}")

print("\n创建验证数据集...")
val_ds = SliceDataset(val_subs, label_map, target_shape=TARGET_SHAPE)
print(f"验证切片数量: {len(val_ds)}")

# 创建数据加载器（优化版：并行加载+GPU传输加速）
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # 并行加载数据
    pin_memory=True,  # 加速GPU传输
    persistent_workers=True  # 保持worker进程活跃
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
# 7. 可视化样本
# ============================================================================

# 获取一个样本
sample_img, sample_mask = train_ds[len(train_ds)//2]

print(f"\n图像形状: {sample_img.shape}")
print(f"掩码形状: {sample_mask.shape}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原始CT图像
axes[0].imshow(sample_img[0], cmap='gray')
axes[0].set_title('CT Image')
axes[0].axis('off')

# 所有掩码叠加
all_masks = sample_mask.sum(dim=0).numpy()
axes[1].imshow(sample_img[0], cmap='gray')
axes[1].imshow(all_masks, alpha=0.3, cmap='jet')
axes[1].set_title('All Masks Overlay')
axes[1].axis('off')

# 单个掩码示例（选择第一个非空掩码）
for i in range(sample_mask.shape[0]):
    if sample_mask[i].sum() > 0:
        axes[2].imshow(sample_img[0], cmap='gray')
        axes[2].imshow(sample_mask[i], alpha=0.5, cmap='Reds')
        axes[2].set_title(f'Sample Mask (channel {i})')
        axes[2].axis('off')
        break

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_visualization_enhanced.png'), dpi=100, bbox_inches='tight')
plt.close()
print("\n样本可视化已保存")


# ============================================================================
# 8. 创建模型
# ============================================================================

# 创建模型
model = UNet2D(in_ch=1, out_ch=len(label_map), features=[32, 64, 128, 256]).to(device)

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
        print(f"\n✓ 使用余弦退火学习率调度器")
        print(f"  学习率范围: {LEARNING_RATE} → {COSINE_ETA_MIN}")
        print(f"  周期: {COSINE_T_MAX} epochs")
    elif SCHEDULER_TYPE == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # 监控val_dice最大化
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE,
            min_lr=PLATEAU_MIN_LR,
            verbose=True
        )
        print(f"\n✓ 使用ReduceLROnPlateau学习率调度器")
        print(f"  初始学习率: {LEARNING_RATE}")
        print(f"  衰减因子: {PLATEAU_FACTOR}")
        print(f"  容忍轮数: {PLATEAU_PATIENCE}")
        print(f"  最小学习率: {PLATEAU_MIN_LR}")
    else:
        scheduler = None
        print(f"\n⚠️  未知的调度器类型: {SCHEDULER_TYPE}，使用固定学习率")
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
# 9. 初始化 Weights & Biases（增强版）
# ============================================================================

if USE_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        notes=WANDB_NOTES,
        config={
            'architecture': 'U-Net 2D',
            'dataset': 'Medical CT Segmentation',
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss': 'BCEWithLogitsLoss',
            'use_scheduler': USE_SCHEDULER,
            'scheduler_type': SCHEDULER_TYPE if USE_SCHEDULER else 'none',
            'cosine_t_max': COSINE_T_MAX if SCHEDULER_TYPE == 'cosine' else None,
            'cosine_eta_min': COSINE_ETA_MIN if SCHEDULER_TYPE == 'cosine' else None,
            'plateau_factor': PLATEAU_FACTOR if SCHEDULER_TYPE == 'plateau' else None,
            'plateau_patience': PLATEAU_PATIENCE if SCHEDULER_TYPE == 'plateau' else None,
            'input_shape': TARGET_SHAPE,
            'num_classes': len(label_map),
            'train_subjects': len(train_subs),
            'val_subjects': len(val_subs),
            'train_slices': len(train_ds),
            'val_slices': len(val_ds),
            'train_ratio': len(train_subs) / n,
            'val_ratio': len(val_subs) / n,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_workers': 4,
            'pin_memory': True,
        }
    )
    # 记录模型
    wandb.watch(model, log='all', log_freq=100)
    print("\n✓ Weights & Biases 已初始化（增强版）")
    print(f"项目: {WANDB_PROJECT}")
    print(f"运行名称: {WANDB_RUN_NAME}")


# ============================================================================
# 10. 训练模型（增强版 - 更多监控指标）
# ============================================================================

# 训练历史记录
history = {
    'train_loss': [],
    'train_dice': [],
    'val_loss': [],
    'val_dice': [],
    'val_iou': [],
    'learning_rate': [],
    'gradient_norm': [],
    'epoch_time': [],
}

print(f"\n开始训练 {EPOCHS} 个epoch...\n")

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()

    print(f"{'='*60}")
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"{'='*60}")

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

        # 实时记录训练指标到 wandb
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
    val_bar = tqdm(val_loader, desc='Validation')

    # 用于记录样本图像和每个类别的dice
    sample_images = []
    sample_masks = []
    sample_preds = []
    all_class_dices = []

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
            val_bar.set_postfix({'dice': f'{dice:.4f}', 'iou': f'{iou:.4f}'})

            # 计算每个类别的dice（用于分析）
            class_dices = dice_score_per_class(preds_bin, masks)
            all_class_dices.append(class_dices.cpu())

            # 保存第一个batch的样本用于可视化
            if batch_idx == 0 and USE_WANDB and epoch % LOG_IMAGES_EVERY_N_EPOCHS == 0:
                sample_images = imgs.cpu()
                sample_masks = masks.cpu()
                sample_preds = preds_bin.cpu()

    avg_val_loss = val_running_loss / len(val_loader)
    avg_val_dice = float(np.mean(val_dice_scores))
    avg_val_iou = float(np.mean(val_iou_scores))

    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)
    history['val_iou'].append(avg_val_iou)

    # 计算每个类别的平均dice
    mean_class_dices = torch.stack(all_class_dices).mean(dim=0)

    # 找出表现最好和最差的5个类别
    top5_indices = torch.topk(mean_class_dices, k=min(5, len(mean_class_dices))).indices
    bottom5_indices = torch.topk(mean_class_dices, k=min(5, len(mean_class_dices)), largest=False).indices

    # 计算epoch时间
    epoch_time = time.time() - epoch_start_time
    history['epoch_time'].append(epoch_time)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])

    # 计算过拟合指标
    overfit_gap = avg_train_dice - avg_val_dice

    # ========== 打印结果 ==========
    print(f"\nEpoch {epoch} 结果:")
    print(f"  训练损失: {avg_train_loss:.4f} | 训练Dice: {avg_train_dice:.4f}")
    print(f"  验证损失: {avg_val_loss:.4f} | 验证Dice: {avg_val_dice:.4f} | 验证IoU: {avg_val_iou:.4f}")
    print(f"  过拟合差距: {overfit_gap:.4f} ({'过拟合' if overfit_gap > 0.1 else '正常'})")
    print(f"  梯度范数: {grad_norm:.4f}")
    print(f"  Epoch耗时: {epoch_time:.1f}秒")

    print(f"\n  表现最好的5个结构:")
    for i, idx in enumerate(top5_indices[:5]):
        print(f"    {idx_to_label[idx.item()]}: {mean_class_dices[idx]:.4f}")

    print(f"\n  表现最差的5个结构:")
    for i, idx in enumerate(bottom5_indices[:5]):
        print(f"    {idx_to_label[idx.item()]}: {mean_class_dices[idx]:.4f}")

    # ========== 记录到 wandb（增强版） ==========
    if USE_WANDB:
        log_dict = {
            'epoch': epoch,
            # 训练指标
            'train/loss': avg_train_loss,
            'train/dice': avg_train_dice,
            # 验证指标
            'val/loss': avg_val_loss,
            'val/dice': avg_val_dice,
            'val/iou': avg_val_iou,
            # 对比指标
            'comparison/loss_gap': avg_val_loss - avg_train_loss,
            'comparison/dice_gap': avg_val_dice - avg_train_dice,
            'comparison/overfit_score': overfit_gap,
            # 训练状态
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_norm': grad_norm,
            'epoch_time': epoch_time,
            'samples_per_second': len(train_ds) / epoch_time,
        }

        # 记录top-5和bottom-5类别的dice
        for i, idx in enumerate(top5_indices[:5]):
            log_dict[f'top_classes/rank_{i+1}_{idx_to_label[idx.item()][:20]}'] = mean_class_dices[idx].item()

        for i, idx in enumerate(bottom5_indices[:5]):
            log_dict[f'bottom_classes/rank_{i+1}_{idx_to_label[idx.item()][:20]}'] = mean_class_dices[idx].item()

        # 记录样本图像（每N个epoch一次）
        if epoch % LOG_IMAGES_EVERY_N_EPOCHS == 0 and len(sample_images) > 0:
            # 选择第一个样本
            img_sample = sample_images[0, 0].numpy()
            mask_sample = sample_masks[0].sum(0).numpy()
            pred_sample = sample_preds[0].sum(0).numpy()

            # 创建对比图像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_sample, cmap='gray')
            axes[0].set_title('CT Image')
            axes[0].axis('off')

            axes[1].imshow(img_sample, cmap='gray')
            axes[1].imshow(mask_sample, alpha=0.5, cmap='Reds')
            axes[1].set_title(f'Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(img_sample, cmap='gray')
            axes[2].imshow(pred_sample, alpha=0.5, cmap='Blues')
            axes[2].set_title(f'Prediction (Dice: {avg_val_dice:.3f})')
            axes[2].axis('off')

            plt.tight_layout()

            log_dict['predictions'] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_dict)

    # ========== 保存检查点 ==========
    checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_enhanced_epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'train_dice': avg_train_dice,
        'val_loss': avg_val_loss,
        'val_dice': avg_val_dice,
        'val_iou': avg_val_iou,
    }, checkpoint_path)
    print(f"  检查点已保存: {checkpoint_path}")

    # 记录最佳模型到 wandb
    if USE_WANDB and avg_val_dice == max(history['val_dice']):
        wandb.run.summary["best_val_dice"] = avg_val_dice
        wandb.run.summary["best_val_iou"] = avg_val_iou
        wandb.run.summary["best_epoch"] = epoch

    # ========== 更新学习率 ==========
    if USE_SCHEDULER and scheduler is not None:
        if SCHEDULER_TYPE == 'cosine':
            # CosineAnnealingLR: 直接step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  学习率已更新: {current_lr:.8f}")
        elif SCHEDULER_TYPE == 'plateau':
            # ReduceLROnPlateau: 根据val_dice自动调整
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_dice)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  学习率已降低: {old_lr:.8f} → {new_lr:.8f}")
            else:
                print(f"  学习率保持: {new_lr:.8f}")

    print()

print(f"\n{'='*60}")
print("训练完成！")
if USE_WANDB:
    print(f"查看完整训练报告: {wandb.run.get_url()}")
print(f"{'='*60}")


# ============================================================================
# 11. 训练历史可视化
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 训练损失 vs 验证损失
axes[0, 0].plot(range(1, EPOCHS + 1), history['train_loss'], marker='o', label='Train Loss', linewidth=2)
axes[0, 0].plot(range(1, EPOCHS + 1), history['val_loss'], marker='s', label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. 训练Dice vs 验证Dice
axes[0, 1].plot(range(1, EPOCHS + 1), history['train_dice'], marker='o', label='Train Dice', linewidth=2, color='green')
axes[0, 1].plot(range(1, EPOCHS + 1), history['val_dice'], marker='s', label='Val Dice', linewidth=2, color='orange')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Dice Score')
axes[0, 1].set_title('Dice Score Comparison')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. 验证IoU
axes[0, 2].plot(range(1, EPOCHS + 1), history['val_iou'], marker='d', linewidth=2, color='purple')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('IoU Score')
axes[0, 2].set_title('Validation IoU')
axes[0, 2].grid(alpha=0.3)

# 4. 过拟合分析（Dice差距）
overfit_gaps = [t - v for t, v in zip(history['train_dice'], history['val_dice'])]
axes[1, 0].plot(range(1, EPOCHS + 1), overfit_gaps, marker='o', linewidth=2, color='red')
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Overfit threshold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Train Dice - Val Dice')
axes[1, 0].set_title('Overfitting Analysis')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 5. 梯度范数
axes[1, 1].plot(range(1, EPOCHS + 1), history['gradient_norm'], marker='o', linewidth=2, color='brown')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Gradient Norm')
axes[1, 1].set_title('Gradient Norm (Stability Check)')
axes[1, 1].grid(alpha=0.3)

# 6. 学习率变化
axes[1, 2].plot(range(1, EPOCHS + 1), history['learning_rate'], marker='o', linewidth=2, color='blue')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Learning Rate')
axes[1, 2].set_title('Learning Rate Schedule')
axes[1, 2].set_yscale('log')  # 使用对数刻度更清楚
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_enhanced.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n训练历史图已保存（增强版）")


# ============================================================================
# 12. 保存训练历史
# ============================================================================

# 保存训练历史为JSON
history_path = os.path.join(OUTPUT_DIR, 'training_history_enhanced.json')
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print(f"训练历史已保存到: {history_path}")

# 打印最终结果摘要
print("\n" + "="*60)
print("训练摘要:")
print(f"  最终训练损失: {history['train_loss'][-1]:.4f}")
print(f"  最终训练Dice: {history['train_dice'][-1]:.4f}")
print(f"  最终验证损失: {history['val_loss'][-1]:.4f}")
print(f"  最终验证Dice: {history['val_dice'][-1]:.4f}")
print(f"  最终验证IoU: {history['val_iou'][-1]:.4f}")
print(f"  最佳验证Dice: {max(history['val_dice']):.4f} (Epoch {history['val_dice'].index(max(history['val_dice'])) + 1})")
print(f"  最终过拟合差距: {history['train_dice'][-1] - history['val_dice'][-1]:.4f}")
print(f"  初始学习率: {history['learning_rate'][0]:.8f}")
print(f"  最终学习率: {history['learning_rate'][-1]:.8f}")
if USE_SCHEDULER:
    print(f"  学习率调度器: {SCHEDULER_TYPE.upper()}")
print(f"  总训练时间: {sum(history['epoch_time']):.1f}秒 ({sum(history['epoch_time'])/60:.1f}分钟)")
print("="*60)


# ============================================================================
# 13. 测试推理
# ============================================================================

# 在验证集上测试一个样本
model.eval()
test_img, test_mask = val_ds[len(val_ds)//2]
test_img_batch = test_img.unsqueeze(0).to(device)

with torch.no_grad():
    test_pred = torch.sigmoid(model(test_img_batch))
    test_pred_bin = (test_pred > 0.5).float()

# 移到CPU用于可视化
test_img = test_img.cpu()
test_mask = test_mask.cpu()
test_pred_bin = test_pred_bin[0].cpu()

# 可视化预测结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原始图像
axes[0, 0].imshow(test_img[0], cmap='gray')
axes[0, 0].set_title('CT Image')
axes[0, 0].axis('off')

# 真实掩码（所有通道）
axes[0, 1].imshow(test_img[0], cmap='gray')
axes[0, 1].imshow(test_mask.sum(dim=0), alpha=0.5, cmap='jet')
axes[0, 1].set_title('Ground Truth (All)')
axes[0, 1].axis('off')

# 预测掩码（所有通道）
axes[0, 2].imshow(test_img[0], cmap='gray')
axes[0, 2].imshow(test_pred_bin.sum(dim=0), alpha=0.5, cmap='jet')
axes[0, 2].set_title('Prediction (All)')
axes[0, 2].axis('off')

# 找三个非空的掩码进行对比
non_empty_channels = [i for i in range(test_mask.shape[0]) if test_mask[i].sum() > 0][:3]

for idx, ch in enumerate(non_empty_channels):
    # 真实掩码
    axes[1, idx].imshow(test_img[0], cmap='gray')
    axes[1, idx].contour(test_mask[ch], colors='red', linewidths=2, alpha=0.7)
    axes[1, idx].contour(test_pred_bin[ch], colors='green', linewidths=2, alpha=0.7, linestyles='dashed')
    axes[1, idx].set_title(f'Channel {ch}\n(Red=GT, Green=Pred)')
    axes[1, idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'test_inference_enhanced.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n测试推理结果已保存（增强版）")

# 计算这个样本的Dice
sample_dice = dice_score(test_pred_bin.unsqueeze(0), test_mask.unsqueeze(0))
print(f"测试样本Dice系数: {sample_dice:.4f}")


# ============================================================================
# 14. 关闭实时监控
# ============================================================================

if USE_WANDB:
    wandb.finish()
    print("\n✓ Weights & Biases 会话已关闭")


# ============================================================================
# 总结
# ============================================================================

print("\n" + "="*60)
print("增强版训练脚本执行完成")
print("\n本脚本的主要特性：")
print("【监控指标】")
print("1. ✓ 训练Dice - 对比训练集和验证集性能")
print("2. ✓ 验证Loss - 查看验证集上的损失")
print("3. ✓ IoU分数 - 另一个常用的分割指标")
print("4. ✓ 梯度范数 - 判断梯度消失/爆炸")
print("5. ✓ Top-5和Bottom-5类别Dice - 看哪些器官分割得最好/最差")
print("6. ✓ 过拟合指标 - Train Dice - Val Dice的差距")
print("7. ✓ 学习率变化曲线 - 实时监控学习率调整")
print("\n【性能优化】")
print("8. ✓ Batch size: 16 (利用24GB显存)")
print("9. ✓ 并行数据加载 (num_workers=4)")
print("10. ✓ GPU传输加速 (pin_memory=True)")
print(f"\n【学习率调度】")
if USE_SCHEDULER:
    print(f"11. ✓ 动态学习率调度器: {SCHEDULER_TYPE.upper()}")
    if SCHEDULER_TYPE == 'cosine':
        print(f"    余弦退火: {LEARNING_RATE} → {COSINE_ETA_MIN}")
    elif SCHEDULER_TYPE == 'plateau':
        print(f"    自适应降低: 性能停滞时自动降低学习率")
else:
    print("11. - 固定学习率")
print("="*60)
