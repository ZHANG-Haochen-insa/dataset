#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环腹部肌肉分割模型训练

利用已训练好的肌肉模型来辅助生成更精确的环腹部肌肉伪标签，然后训练专门的分割模型。

方法：
1. 使用已有U-Net模型分割10种特定肌肉
2. 使用HU阈值检测所有肌肉组织
3. 排除已分割的特定肌肉，得到环腹部肌肉区域
4. 使用这些伪标签训练新模型

环腹部肌肉包括:
- 腹直肌 (Rectus Abdominis)
- 腹外斜肌 (External Oblique)
- 腹内斜肌 (Internal Oblique)
- 腹横肌 (Transversus Abdominis)

作者: hzhang02
日期: 2024-12-17
"""

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
from skimage import morphology
from skimage.measure import label, regionprops
from scipy import ndimage

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

import wandb
import matplotlib.pyplot as plt

os.environ['WANDB_API_KEY'] = 'e331d01a3e6f2b0b78c22ffde1e676cb4742f891'

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# 1. 常量定义
# ============================================================================

MUSCLE_HU_MIN = -29
MUSCLE_HU_MAX = 150
BODY_HU_MIN = -500

# ============================================================================
# 2. U-Net模型定义
# ============================================================================

class DoubleConv(nn.Module):
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
    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        rev = list(reversed(features))
        up_in = features[-1] * 2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(up_in, f))
            up_in = f

        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            trans = self.ups[idx]
            conv = self.ups[idx + 1]
            x = trans(x)
            skip = skips[-(idx // 2) - 1]
            if x.shape != skip.shape:
                _, _, h, w = x.shape
                skip = skip[:, :, :h, :w]
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.final(x)


# ============================================================================
# 3. 加载已有肌肉模型
# ============================================================================

print("\n加载已训练的肌肉模型...")
PRETRAINED_MODEL_PATH = '/local/hzhang02/data/dataset/outputs/best_model.pth'
LABEL_MAP_PATH = '/local/hzhang02/data/dataset/outputs/label_map.json'

with open(LABEL_MAP_PATH, 'r') as f:
    muscle_label_map = json.load(f)
num_muscle_classes = len(muscle_label_map)
print(f"已有模型分割 {num_muscle_classes} 种特定肌肉")

pretrained_model = UNet2D(in_ch=1, out_ch=num_muscle_classes, features=[32, 64, 128, 256])
checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
pretrained_model.load_state_dict(checkpoint['model_state_dict'])
pretrained_model.to(device)
pretrained_model.eval()
print(f"已有肌肉模型加载成功: {PRETRAINED_MODEL_PATH}")


# ============================================================================
# 4. 环腹部肌肉伪标签生成
# ============================================================================

def get_body_mask(slice_img, threshold=-500):
    body = slice_img > threshold
    body = morphology.binary_closing(body, morphology.disk(3))
    body = morphology.binary_opening(body, morphology.disk(2))
    body = ndimage.binary_fill_holes(body)
    labeled = label(body)
    if labeled.max() > 0:
        regions = regionprops(labeled)
        largest = max(regions, key=lambda x: x.area)
        body = labeled == largest.label
    return body.astype(np.float32)


def predict_specific_muscles(slice_normalized, target_shape=(256, 256)):
    """使用已有模型预测特定肌肉"""
    slice_resized = resize(slice_normalized, target_shape, order=1,
                           preserve_range=True, anti_aliasing=True)
    img_t = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = torch.sigmoid(pretrained_model(img_t))
        pred_bin = (pred > 0.5).float()

    # 合并所有特定肌肉
    combined = pred_bin[0].sum(dim=0).cpu().numpy()
    return (combined > 0.5).astype(np.float32)


def generate_abdominal_wall_label(slice_img, original_shape):
    """
    生成环腹部肌肉伪标签

    1. 用已有模型分割特定肌肉
    2. 用HU阈值检测所有肌肉
    3. 排除特定肌肉，定位腹壁区域
    """
    target_shape = (256, 256)

    # 归一化
    lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
    slice_normalized = np.clip(slice_img, lo, hi)
    if hi - lo > 0:
        slice_normalized = (slice_normalized - lo) / (hi - lo)
    else:
        slice_normalized = np.zeros_like(slice_normalized)

    # 1. 预测特定肌肉（使用已有模型）
    specific_muscles = predict_specific_muscles(slice_normalized, target_shape)
    specific_muscles_resized = resize(specific_muscles, original_shape, order=0,
                                       preserve_range=True, anti_aliasing=False)

    # 2. 获取身体掩码和所有肌肉组织
    body_mask = get_body_mask(slice_img)
    all_muscle = (slice_img >= MUSCLE_HU_MIN) & (slice_img <= MUSCLE_HU_MAX)
    all_muscle = all_muscle & body_mask.astype(bool)

    # 3. 定位腹壁区域（皮下到腹腔之间）
    subcutaneous_depth = 15
    abdominal_cavity_depth = 40

    outer_boundary = morphology.binary_erosion(body_mask, morphology.disk(subcutaneous_depth))
    inner_core = morphology.binary_erosion(body_mask, morphology.disk(abdominal_cavity_depth))
    abdominal_wall_region = outer_boundary.astype(bool) & ~inner_core

    # 4. 环腹部肌肉 = 腹壁区域内的肌肉 - 已分割的特定肌肉
    abdominal_wall_muscle = all_muscle & abdominal_wall_region
    abdominal_wall_muscle = abdominal_wall_muscle & ~(specific_muscles_resized > 0.5)

    # 5. 形态学后处理
    abdominal_wall_muscle = morphology.binary_opening(abdominal_wall_muscle, morphology.disk(2))
    abdominal_wall_muscle = morphology.binary_closing(abdominal_wall_muscle, morphology.disk(3))

    # 移除小区域
    labeled = label(abdominal_wall_muscle)
    if labeled.max() > 0:
        for region in regionprops(labeled):
            if region.area < 50:
                abdominal_wall_muscle[labeled == region.label] = 0

    return abdominal_wall_muscle.astype(np.float32)


print("环腹部肌肉伪标签生成函数定义完成（使用已有模型辅助）")


# ============================================================================
# 5. 数据集类
# ============================================================================

class AbdominalWallDataset(Dataset):
    def __init__(self, subjects, target_shape=(256, 256)):
        self.items = []
        self.target_shape = target_shape

        print("构建环腹部肌肉数据集...")
        for s in tqdm(subjects, desc="加载受试者"):
            ct_path = os.path.join(s, 'ct.nii.gz')
            if not os.path.exists(ct_path):
                continue

            img = nib.load(ct_path)
            depth = img.shape[2]

            for z in range(depth):
                self.items.append((ct_path, z))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ct_path, z = self.items[idx]

        img = nib.load(ct_path).get_fdata().astype(np.float32)
        slice_img = img[:, :, z]
        original_shape = slice_img.shape

        # 生成伪标签（使用已有模型辅助）
        abdominal_wall_mask = generate_abdominal_wall_label(slice_img, original_shape)

        # 归一化CT图像
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
        abdominal_wall_mask = resize(abdominal_wall_mask, (H, W), order=0,
                                      preserve_range=True, anti_aliasing=False)
        abdominal_wall_mask = (abdominal_wall_mask > 0.5).astype(np.float32)

        img_t = torch.from_numpy(slice_normalized).unsqueeze(0).float()
        mask_t = torch.from_numpy(abdominal_wall_mask).unsqueeze(0).float()

        return img_t, mask_t


# ============================================================================
# 6. 评估指标
# ============================================================================

def dice_score(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


def iou_score(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


# ============================================================================
# 7. 配置
# ============================================================================

DATA_ROOT = '/local/hzhang02/data'
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs_abdominal_wall'
TARGET_SHAPE = (256, 256)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10
NUM_SUBJECTS = 50

USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 3

USE_WANDB = True
WANDB_PROJECT = 'abdominal-wall-muscle-unet'
WANDB_RUN_NAME = 'unet-2d-abdominal-wall'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("配置信息（环腹部肌肉分割）:")
print(f"  数据根目录: {DATA_ROOT}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  目标结构: 环腹部肌肉（腹壁肌群）")
print(f"  使用已有模型: {PRETRAINED_MODEL_PATH}")
print(f"  批次大小: {BATCH_SIZE}")
print(f"  学习率: {LEARNING_RATE}")
print(f"  最大训练轮数: {EPOCHS}")
print(f"  受试者数量: {NUM_SUBJECTS}")
print("=" * 60)

# 查找受试者
all_subjects_raw = [d for d in os.listdir(DATA_ROOT) if d.startswith('s') and os.path.isdir(os.path.join(DATA_ROOT, d))]
all_subjects = [s for s in all_subjects_raw if s in [f's{i:04d}' for i in range(NUM_SUBJECTS)]]
subjects = [os.path.join(DATA_ROOT, s) for s in sorted(all_subjects)]

print(f"\n找到 {len(subjects)} 个受试者")

# 划分数据集
random.shuffle(subjects)
n = len(subjects)
ntrain = max(1, int(n * 0.8))
train_subs = subjects[:ntrain]
val_subs = subjects[ntrain:]

print(f"训练集: {len(train_subs)} / 验证集: {len(val_subs)}")

# 创建数据集
print("\n创建训练数据集...")
train_ds = AbdominalWallDataset(train_subs, target_shape=TARGET_SHAPE)
print(f"训练切片数量: {len(train_ds)}")

print("\n创建验证数据集...")
val_ds = AbdominalWallDataset(val_subs, target_shape=TARGET_SHAPE)
print(f"验证切片数量: {len(val_ds)}")

# 注意：num_workers=0 因为在__getitem__中使用CUDA推理生成伪标签
# CUDA不能在fork的子进程中重新初始化
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)


# ============================================================================
# 8. 创建新模型
# ============================================================================

# 新模型用于环腹部肌肉分割（单通道输出）
model = UNet2D(in_ch=1, out_ch=1, features=[32, 64, 128, 256]).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n新模型参数量: {total_params:,}")


# ============================================================================
# 9. 初始化 WandB
# ============================================================================

if USE_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            'architecture': 'U-Net 2D',
            'task': 'Abdominal Wall Muscle Segmentation',
            'pretrained_model': PRETRAINED_MODEL_PATH,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_subjects': len(subjects),
        }
    )
    wandb.watch(model, log='all', log_freq=100)
    print("\nWandB 已初始化")


# ============================================================================
# 10. 训练
# ============================================================================

history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
best_val_dice = 0.0
epochs_no_improve = 0

print(f"\n开始训练...\n")

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    print(f"{'=' * 60}")
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"{'=' * 60}")

    # 训练
    model.train()
    train_loss = 0.0
    train_dice_scores = []

    for imgs, masks in tqdm(train_loader, desc='Training'):
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds_bin = (torch.sigmoid(preds) > 0.5).float()
            dice = dice_score(preds_bin, masks).item()
            train_dice_scores.append(dice)

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = np.mean(train_dice_scores)
    history['train_loss'].append(avg_train_loss)
    history['train_dice'].append(avg_train_dice)

    # 验证
    model.eval()
    val_loss = 0.0
    val_dice_scores = []
    val_iou_scores = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc='Validation'):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)
            val_loss += loss.item()

            preds_bin = (torch.sigmoid(preds) > 0.5).float()
            dice = dice_score(preds_bin, masks).item()
            iou = iou_score(preds_bin, masks).item()
            val_dice_scores.append(dice)
            val_iou_scores.append(iou)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = np.mean(val_dice_scores)
    avg_val_iou = np.mean(val_iou_scores)
    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)
    history['val_iou'].append(avg_val_iou)

    epoch_time = time.time() - epoch_start

    print(f"\n结果:")
    print(f"  训练损失: {avg_train_loss:.4f} | 训练Dice: {avg_train_dice:.4f}")
    print(f"  验证损失: {avg_val_loss:.4f} | 验证Dice: {avg_val_dice:.4f} | 验证IoU: {avg_val_iou:.4f}")
    print(f"  耗时: {epoch_time:.1f}秒")

    # WandB记录
    if USE_WANDB:
        wandb.log({
            'epoch': epoch,
            'train/loss': avg_train_loss,
            'train/dice': avg_train_dice,
            'val/loss': avg_val_loss,
            'val/dice': avg_val_dice,
            'val/iou': avg_val_iou,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

    # 保存检查点
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': avg_val_dice,
    }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch{epoch}.pth'))

    # 最佳模型
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_dice': avg_val_dice,
        }, os.path.join(OUTPUT_DIR, 'best_abdominal_wall_model.pth'))
        print(f"  新最佳模型！Dice: {best_val_dice:.4f}")
    else:
        epochs_no_improve += 1
        print(f"  无改善 ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

    scheduler.step()

    # 早停
    if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n早停触发！")
        break

print(f"\n{'=' * 60}")
print("训练完成！")
print(f"最佳验证Dice: {best_val_dice:.4f}")
print(f"{'=' * 60}")

# 保存历史
with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

if USE_WANDB:
    wandb.finish()

print(f"\n结果已保存到: {OUTPUT_DIR}")
