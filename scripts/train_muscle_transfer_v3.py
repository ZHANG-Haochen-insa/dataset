#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肌肉特征迁移学习 V3 - 先扩后缩方法

核心思想：
1. 先扩：用HU值范围做粗分割，覆盖所有可能是肌肉的区域（高召回）
2. 后缩：用已标注肌肉区域作为参考，学习如何精修边界

损失函数设计：
- 已标注区域：必须与标签高度一致（边界对齐）
- HU合理但未标注区域：鼓励覆盖，不惩罚扩展
- HU不合理区域：强制排除（空气/骨骼）

作者: hzhang02
日期: 2026-01-11
"""

import os
import json
import random
import time
from typing import List, Dict
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import wandb

# 设置随机种子
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

# 肌肉HU值范围
MUSCLE_HU_MIN = -29
MUSCLE_HU_MAX = 150

# 硬排除区域
AIR_HU_MAX = -200
BONE_HU_MIN = 300
BODY_HU_MIN = -500

# 已知肌肉分割文件
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

# 腹部区域标记
ABDOMINAL_ORGANS = [
    'liver.nii.gz', 'spleen.nii.gz', 'kidney_left.nii.gz', 'kidney_right.nii.gz',
    'pancreas.nii.gz', 'stomach.nii.gz', 'colon.nii.gz',
]
EXCLUDE_MARKERS = ['femur_left.nii.gz', 'femur_right.nii.gz']

# 训练超参数
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
EPOCHS = 30

# 可视化
VIS_EVERY_N_EPOCHS = 2
VIS_NUM_SAMPLES = 4

# 损失权重 - 关键调整
LABEL_ALIGNMENT_WEIGHT = 3.0    # 已标注区域边界对齐（高权重）
COVERAGE_REWARD_WEIGHT = 1.0    # 覆盖奖励（鼓励扩展）
EXCLUSION_WEIGHT = 5.0          # 排除区域约束
SMOOTHNESS_WEIGHT = 0.2         # 边界平滑

# 是否只用腹部
ABDOMINAL_ONLY = True


# ============================================================================
# 2. 网络结构
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MuscleRefineNet(nn.Module):
    """
    肌肉精修网络

    输入通道：
    - CT图像（归一化）
    - HU粗分割（所有HU在肌肉范围内的像素）
    - 已知肌肉标签
    - 排除区域mask

    任务：从HU粗分割中学习如何精修边界
    """

    def __init__(self, in_ch=4, features=[32, 64, 128, 256]):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        current_ch = in_ch
        for f in features:
            self.encoder.append(DoubleConv(current_ch, f, dropout=0.1))
            current_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=0.2)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        rev_features = list(reversed(features))
        up_ch = features[-1] * 2

        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(up_ch, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f, dropout=0.1))
            up_ch = f

        self.final = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final(x)


# ============================================================================
# 3. 损失函数 - 先扩后缩
# ============================================================================

class ExpandThenRefireLoss(nn.Module):
    """
    先扩后缩损失函数

    核心思想：
    1. 已标注区域：必须精确对齐（高权重BCE）
    2. HU合理但未标注区域：鼓励覆盖（不惩罚预测为肌肉）
    3. HU不合理区域：强制排除
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, label_mask, hu_coarse, exclusion_mask, body_mask):
        """
        Args:
            pred_logits: 模型预测 (B, 1, H, W)
            label_mask: 已知肌肉标签 (B, 1, H, W)
            hu_coarse: HU粗分割 (B, 1, H, W)，所有HU在肌肉范围内的像素
            exclusion_mask: 排除区域 (B, 1, H, W)
            body_mask: 身体掩码 (B, 1, H, W)
        """
        pred_prob = torch.sigmoid(pred_logits)
        losses = {}

        # ============ 1. 已标注区域对齐损失 ============
        # 在已标注区域，预测必须与标签高度一致
        has_label = (label_mask > 0.5).float()

        # BCE损失，但只在有标签的区域计算
        bce_all = self.bce(pred_logits, label_mask)

        # 有标签区域权重更高
        label_region_loss = (bce_all * has_label).sum() / (has_label.sum() + 1e-6)
        losses['label_alignment'] = label_region_loss * LABEL_ALIGNMENT_WEIGHT

        # ============ 2. 覆盖奖励 ============
        # 在HU合理区域（hu_coarse=1）预测为肌肉是好的，不应该惩罚
        # 只惩罚在HU合理区域内漏检（预测为0但HU合理）

        # 未标注但HU合理的区域
        unlabeled_hu_valid = hu_coarse * (1 - has_label) * body_mask * (1 - exclusion_mask)

        # 在这些区域，我们希望模型预测为肌肉（覆盖）
        # 所以如果预测概率低，给予惩罚（鼓励覆盖）
        # 但惩罚较轻，因为这些区域没有真实标签
        coverage_loss = ((1 - pred_prob) * unlabeled_hu_valid).mean()
        losses['coverage_reward'] = coverage_loss * COVERAGE_REWARD_WEIGHT

        # ============ 3. 排除区域损失 ============
        # 空气/骨骼区域必须预测为0
        if exclusion_mask.sum() > 0:
            exclusion_loss = (pred_prob * exclusion_mask).mean()
        else:
            exclusion_loss = torch.tensor(0.0, device=pred_logits.device)
        losses['exclusion'] = exclusion_loss * EXCLUSION_WEIGHT

        # ============ 4. HU范围外惩罚 ============
        # 在HU不在肌肉范围内的区域（但不是排除区域），轻微惩罚
        hu_invalid = (1 - hu_coarse) * body_mask * (1 - exclusion_mask)
        hu_violation = (pred_prob * hu_invalid).mean()
        losses['hu_violation'] = hu_violation * 1.0

        # ============ 5. 边界平滑 ============
        tv_h = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]).mean()
        tv_w = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]).mean()
        losses['smoothness'] = (tv_h + tv_w) * SMOOTHNESS_WEIGHT

        total_loss = sum(losses.values())
        losses['total'] = total_loss

        return total_loss, losses


# ============================================================================
# 4. 数据集
# ============================================================================

class MuscleRefineDataset(Dataset):
    """
    肌肉精修数据集

    提供：
    - CT图像
    - HU粗分割（所有HU在肌肉范围内的像素）
    - 已知肌肉标签
    - 排除区域
    """

    def __init__(self, subjects, target_shape=(256, 256), abdominal_only=True):
        self.items = []
        self.target_shape = target_shape
        self.muscle_files = KNOWN_MUSCLE_FILES
        self.abdominal_only = abdominal_only

        print(f"构建数据集... (腹部筛选: {abdominal_only})")
        total_slices = 0
        kept_slices = 0

        for s in tqdm(subjects, desc="扫描受试者"):
            ct_path = os.path.join(s, 'ct.nii.gz')
            seg_dir = os.path.join(s, 'segmentations')

            if not os.path.exists(ct_path) or not os.path.exists(seg_dir):
                continue

            has_muscle = any(os.path.exists(os.path.join(seg_dir, f)) for f in self.muscle_files)
            if not has_muscle:
                continue

            img = nib.load(ct_path)
            depth = img.shape[2]

            if self.abdominal_only:
                abdominal_masks = []
                for organ in ABDOMINAL_ORGANS:
                    organ_path = os.path.join(seg_dir, organ)
                    if os.path.exists(organ_path):
                        abdominal_masks.append(nib.load(organ_path).get_fdata())

                exclude_masks = []
                for marker in EXCLUDE_MARKERS:
                    marker_path = os.path.join(seg_dir, marker)
                    if os.path.exists(marker_path):
                        exclude_masks.append(nib.load(marker_path).get_fdata())

            for z in range(depth):
                total_slices += 1

                if self.abdominal_only:
                    has_abdominal = any(
                        mask[:, :, z].sum() > 0 for mask in abdominal_masks
                    ) if abdominal_masks else False

                    has_femur = any(
                        mask[:, :, z].sum() > 0 for mask in exclude_masks
                    ) if exclude_masks else False

                    if has_femur:
                        continue
                    if not has_abdominal and not abdominal_masks:
                        continue

                self.items.append((s, z))
                kept_slices += 1

        if self.abdominal_only:
            print(f"   腹部筛选: {kept_slices}/{total_slices} 切片保留 ({100*kept_slices/max(total_slices,1):.1f}%)")

    def __len__(self):
        return len(self.items)

    def _load_muscle_mask(self, subject_dir, z_idx, original_shape):
        seg_dir = os.path.join(subject_dir, 'segmentations')
        combined_mask = np.zeros(original_shape, dtype=np.float32)

        for muscle_file in self.muscle_files:
            seg_path = os.path.join(seg_dir, muscle_file)
            if os.path.exists(seg_path):
                seg = nib.load(seg_path).get_fdata()
                if z_idx < seg.shape[2]:
                    combined_mask = np.maximum(combined_mask, seg[:, :, z_idx])

        return (combined_mask > 0.5).astype(np.float32)

    def _get_body_mask(self, hu_slice):
        body = hu_slice > BODY_HU_MIN
        body = morphology.binary_closing(body, morphology.disk(3))
        body = morphology.binary_opening(body, morphology.disk(2))
        body = ndimage.binary_fill_holes(body)
        labeled = label(body)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            largest = max(regions, key=lambda x: x.area)
            body = labeled == largest.label
        return body.astype(np.float32)

    def __getitem__(self, idx):
        subject_dir, z = self.items[idx]

        ct_path = os.path.join(subject_dir, 'ct.nii.gz')
        img = nib.load(ct_path).get_fdata().astype(np.float32)
        hu_slice = img[:, :, z]
        original_shape = hu_slice.shape[:2]

        # 已知肌肉标签
        muscle_mask = self._load_muscle_mask(subject_dir, z, original_shape)

        # 归一化CT
        lo, hi = np.percentile(hu_slice, 1), np.percentile(hu_slice, 99)
        if hi - lo > 0:
            ct_normalized = np.clip(hu_slice, lo, hi)
            ct_normalized = (ct_normalized - lo) / (hi - lo)
        else:
            ct_normalized = np.zeros_like(hu_slice)

        # 调整大小
        H, W = self.target_shape
        ct_resized = resize(ct_normalized, (H, W), order=1, preserve_range=True)
        hu_resized = resize(hu_slice, (H, W), order=1, preserve_range=True)
        muscle_resized = resize(muscle_mask, (H, W), order=0, preserve_range=True)

        # HU粗分割：所有HU在肌肉范围内的像素
        hu_coarse = ((hu_resized >= MUSCLE_HU_MIN) & (hu_resized <= MUSCLE_HU_MAX)).astype(np.float32)

        # 排除区域
        exclusion = ((hu_resized < AIR_HU_MAX) | (hu_resized > BONE_HU_MIN)).astype(np.float32)

        # 身体掩码
        body_mask = self._get_body_mask(hu_resized)

        # 构建输入 (4通道: CT + HU粗分割 + 已知标签 + 非排除区域)
        input_tensor = np.stack([
            ct_resized,           # CT图像
            hu_coarse,            # HU粗分割（候选区域）
            muscle_resized,       # 已知肌肉标签（参考）
            1 - exclusion         # 非排除区域
        ], axis=0)

        input_t = torch.from_numpy(input_tensor).float()
        hu_coarse_t = torch.from_numpy(hu_coarse).unsqueeze(0).float()
        muscle_t = torch.from_numpy(muscle_resized).unsqueeze(0).float()
        body_t = torch.from_numpy(body_mask).unsqueeze(0).float()
        exclusion_t = torch.from_numpy(exclusion).unsqueeze(0).float()

        return input_t, hu_coarse_t, muscle_t, body_t, exclusion_t


# ============================================================================
# 5. 评估指标
# ============================================================================

def compute_metrics(pred, label, hu_coarse, body_mask, exclusion_mask):
    pred_binary = (pred > 0.5).float()
    label_binary = (label > 0.5).float()

    metrics = {}

    # 1. 标签区域覆盖率（召回率）- 关键指标
    if label_binary.sum() > 0:
        recall = (pred_binary * label_binary).sum() / label_binary.sum()
        metrics['label_recall'] = recall.item()
    else:
        metrics['label_recall'] = 1.0

    # 2. 标签区域Dice
    if label_binary.sum() > 0:
        inter = (pred_binary * label_binary).sum()
        union = pred_binary.sum() + label_binary.sum()
        dice = (2 * inter) / (union + 1e-6)
        metrics['label_dice'] = dice.item()
    else:
        metrics['label_dice'] = 1.0

    # 3. HU粗分割覆盖率（预测覆盖了多少HU合理区域）
    hu_valid_region = hu_coarse * body_mask * (1 - exclusion_mask)
    if hu_valid_region.sum() > 0:
        hu_coverage = (pred_binary * hu_valid_region).sum() / hu_valid_region.sum()
        metrics['hu_coverage'] = hu_coverage.item()
    else:
        metrics['hu_coverage'] = 0.0

    # 4. 排除区域溢出
    if exclusion_mask.sum() > 0:
        overflow = (pred_binary * exclusion_mask).sum() / exclusion_mask.sum()
        metrics['exclusion_overflow'] = overflow.item()
    else:
        metrics['exclusion_overflow'] = 0.0

    # 5. 扩展比例
    if label_binary.sum() > 0:
        pred_in_body = pred_binary * body_mask
        expansion = pred_in_body.sum() / label_binary.sum()
        metrics['expansion_ratio'] = expansion.item()
    else:
        metrics['expansion_ratio'] = 1.0

    return metrics


# ============================================================================
# 6. 可视化
# ============================================================================

def visualize_predictions(model, dataloader, device, epoch, output_dir, num_samples=4):
    model.eval()

    samples = []
    with torch.no_grad():
        for inputs, hu_coarse, labels, body_masks, exclusion_masks in dataloader:
            if len(samples) >= num_samples:
                break

            inputs = inputs.to(device)
            pred_logits = model(inputs)
            pred_prob = torch.sigmoid(pred_logits)

            for i in range(min(inputs.size(0), num_samples - len(samples))):
                samples.append({
                    'ct': inputs[i, 0].cpu().numpy(),
                    'hu_coarse': inputs[i, 1].cpu().numpy(),
                    'label': labels[i, 0].numpy(),
                    'pred': pred_prob[i, 0].cpu().numpy(),
                    'exclusion': exclusion_masks[i, 0].numpy(),
                })
                if len(samples) >= num_samples:
                    break

    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample in enumerate(samples):
        ct = sample['ct']
        hu_coarse = sample['hu_coarse']
        label = sample['label']
        pred = sample['pred']
        pred_binary = (pred > 0.5).astype(np.float32)
        exclusion = sample['exclusion']

        # 1. CT原图
        axes[idx, 0].imshow(ct, cmap='gray')
        axes[idx, 0].set_title('CT Image')
        axes[idx, 0].axis('off')

        # 2. HU粗分割（候选区域）
        axes[idx, 1].imshow(ct, cmap='gray')
        axes[idx, 1].imshow(hu_coarse, cmap='Greens', alpha=0.4)
        axes[idx, 1].set_title('HU Coarse (Candidate)')
        axes[idx, 1].axis('off')

        # 3. 已知标签
        axes[idx, 2].imshow(ct, cmap='gray')
        axes[idx, 2].imshow(label, cmap='Reds', alpha=0.5)
        axes[idx, 2].set_title('Known Labels')
        axes[idx, 2].axis('off')

        # 4. 模型预测
        axes[idx, 3].imshow(ct, cmap='gray')
        axes[idx, 3].imshow(pred, cmap='Blues', alpha=0.5)
        axes[idx, 3].set_title('Prediction')
        axes[idx, 3].axis('off')

        # 5. 对比：预测 vs 标签
        overlay = np.zeros((*ct.shape, 3))
        overlay[..., 0] = label * (1 - pred_binary)  # 红：漏检
        overlay[..., 1] = pred_binary * (1 - label)  # 绿：新覆盖
        overlay[..., 2] = pred_binary * label        # 蓝：正确匹配
        axes[idx, 4].imshow(ct, cmap='gray')
        axes[idx, 4].imshow(overlay, alpha=0.6)
        axes[idx, 4].set_title('R:Miss G:New B:Match')
        axes[idx, 4].axis('off')

        # 6. 预测 vs HU粗分割
        overlay2 = np.zeros((*ct.shape, 3))
        overlay2[..., 0] = hu_coarse * (1 - pred_binary)  # 红：HU合理但未预测
        overlay2[..., 1] = pred_binary * hu_coarse        # 绿：HU合理且预测
        overlay2[..., 2] = pred_binary * (1 - hu_coarse)  # 蓝：HU不合理但预测（问题）
        axes[idx, 5].imshow(ct, cmap='gray')
        axes[idx, 5].imshow(overlay2, alpha=0.6)
        axes[idx, 5].set_title('R:Missed G:Good B:Bad')
        axes[idx, 5].axis('off')

    plt.suptitle(f'Epoch {epoch} - Expand then Refine', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'visualization_epoch{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return wandb.Image(save_path, caption=f"Epoch {epoch}")


# ============================================================================
# 7. 主训练流程
# ============================================================================

def main():
    print("=" * 60)
    print("肌肉特征迁移学习 V3 - 先扩后缩方法")
    print("=" * 60)

    DATA_ROOT = '/local/hzhang02/data'
    OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs_muscle_transfer_v3'
    TARGET_SHAPE = (256, 256)
    NUM_SUBJECTS = 50

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # wandb
    wandb.init(
        project="muscle-transfer-learning",
        name=f"transfer-v3-expand-refine-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "version": "v3-expand-refine",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "label_alignment_weight": LABEL_ALIGNMENT_WEIGHT,
            "coverage_reward_weight": COVERAGE_REWARD_WEIGHT,
            "exclusion_weight": EXCLUSION_WEIGHT,
            "smoothness_weight": SMOOTHNESS_WEIGHT,
            "muscle_hu_range": [MUSCLE_HU_MIN, MUSCLE_HU_MAX],
            "abdominal_only": ABDOMINAL_ONLY,
        }
    )
    print("   wandb 初始化完成")

    # 模型
    print("\n1. 初始化模型...")
    model = MuscleRefineNet(in_ch=4, features=[32, 64, 128, 256])
    model.to(device)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = ExpandThenRefireLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 数据
    print("\n2. 加载训练数据...")
    all_subjects_raw = [d for d in os.listdir(DATA_ROOT)
                        if d.startswith('s') and os.path.isdir(os.path.join(DATA_ROOT, d))]
    all_subjects = [s for s in all_subjects_raw if s in [f's{i:04d}' for i in range(NUM_SUBJECTS)]]
    subjects = [os.path.join(DATA_ROOT, s) for s in sorted(all_subjects)]

    print(f"   找到 {len(subjects)} 个受试者")

    random.shuffle(subjects)
    n = len(subjects)
    ntrain = max(1, int(n * 0.8))
    train_subs = subjects[:ntrain]
    val_subs = subjects[ntrain:]

    print(f"   训练集: {len(train_subs)} / 验证集: {len(val_subs)}")

    train_ds = MuscleRefineDataset(train_subs, TARGET_SHAPE, abdominal_only=ABDOMINAL_ONLY)
    val_ds = MuscleRefineDataset(val_subs, TARGET_SHAPE, abdominal_only=ABDOMINAL_ONLY)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"   训练切片: {len(train_ds)} / 验证切片: {len(val_ds)}")

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 训练
    print("\n3. 开始训练...")
    print("=" * 60)

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'label_recall': [], 'label_dice': [], 'hu_coverage': [],
        'exclusion_overflow': [], 'expansion_ratio': []
    }

    best_recall = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)

        # 训练
        model.train()
        train_losses = []
        loss_components = {'label_alignment': [], 'coverage_reward': [], 'exclusion': [], 'hu_violation': [], 'smoothness': []}

        for inputs, hu_coarse, labels, body_masks, exclusion_masks in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            hu_coarse = hu_coarse.to(device)
            labels = labels.to(device)
            body_masks = body_masks.to(device)
            exclusion_masks = exclusion_masks.to(device)

            optimizer.zero_grad()
            pred_logits = model(inputs)
            loss, loss_dict = criterion(pred_logits, labels, hu_coarse, exclusion_masks, body_masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            for k in loss_components:
                if k in loss_dict:
                    loss_components[k].append(loss_dict[k].item())

        # 验证
        model.eval()
        val_losses = []
        val_metrics = {'label_recall': [], 'label_dice': [], 'hu_coverage': [],
                       'exclusion_overflow': [], 'expansion_ratio': []}

        with torch.no_grad():
            for inputs, hu_coarse, labels, body_masks, exclusion_masks in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                hu_coarse = hu_coarse.to(device)
                labels = labels.to(device)
                body_masks = body_masks.to(device)
                exclusion_masks = exclusion_masks.to(device)

                pred_logits = model(inputs)
                loss, _ = criterion(pred_logits, labels, hu_coarse, exclusion_masks, body_masks)
                val_losses.append(loss.item())

                pred_prob = torch.sigmoid(pred_logits)
                for i in range(inputs.size(0)):
                    metrics = compute_metrics(
                        pred_prob[i], labels[i], hu_coarse[i],
                        body_masks[i], exclusion_masks[i]
                    )
                    for k, v in metrics.items():
                        val_metrics[k].append(v)

        scheduler.step()

        # 记录
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        for k, v in avg_metrics.items():
            history[k].append(v)

        epoch_time = time.time() - epoch_start

        print(f"\n结果:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  标签召回率: {avg_metrics['label_recall']:.4f} (越高越好)")
        print(f"  标签Dice: {avg_metrics['label_dice']:.4f}")
        print(f"  HU覆盖率: {avg_metrics['hu_coverage']:.4f}")
        print(f"  排除溢出: {avg_metrics['exclusion_overflow']:.4f}")
        print(f"  扩展比例: {avg_metrics['expansion_ratio']:.2f}x")
        print(f"  耗时: {epoch_time:.1f}秒")

        # wandb
        log_dict = {
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/label_alignment": np.mean(loss_components['label_alignment']),
            "train/coverage_reward": np.mean(loss_components['coverage_reward']),
            "train/exclusion": np.mean(loss_components['exclusion']),
            "val/loss": avg_val_loss,
            "val/label_recall": avg_metrics['label_recall'],
            "val/label_dice": avg_metrics['label_dice'],
            "val/hu_coverage": avg_metrics['hu_coverage'],
            "val/exclusion_overflow": avg_metrics['exclusion_overflow'],
            "val/expansion_ratio": avg_metrics['expansion_ratio'],
            "lr": scheduler.get_last_lr()[0],
        }

        if epoch % VIS_EVERY_N_EPOCHS == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  生成可视化...")
            vis_image = visualize_predictions(model, val_loader, device, epoch, OUTPUT_DIR, VIS_NUM_SAMPLES)
            log_dict["predictions"] = vis_image

        wandb.log(log_dict)

        # 保存
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': avg_metrics,
        }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch{epoch}.pth'))

        # 最佳模型（基于召回率）
        if avg_metrics['label_recall'] > best_recall:
            best_recall = avg_metrics['label_recall']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': avg_metrics,
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  ★ 新最佳模型！召回率: {best_recall:.4f}")

    # 保存结果
    print("\n4. 保存训练结果...")

    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 可视化曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['epoch'], history['label_recall'])
    axes[0, 1].set_title('Label Recall (↑)')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['epoch'], history['label_dice'])
    axes[0, 2].set_title('Label Dice (↑)')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history['epoch'], history['hu_coverage'])
    axes[1, 0].set_title('HU Coverage (↑)')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['epoch'], history['exclusion_overflow'])
    axes[1, 1].set_title('Exclusion Overflow (↓)')
    axes[1, 1].set_ylim([0, 0.5])
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history['epoch'], history['expansion_ratio'])
    axes[1, 2].set_title('Expansion Ratio')
    axes[1, 2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
    plt.close()

    wandb.log({"training_curves": wandb.Image(os.path.join(OUTPUT_DIR, 'training_history.png'))})
    wandb.summary["best_recall"] = best_recall
    wandb.finish()

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  最佳召回率: {best_recall:.4f}")
    print(f"  结果保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
