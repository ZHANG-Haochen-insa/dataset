#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肌肉特征迁移学习 V2 - 基于TotalSegmentator标签

改进：
1. 直接使用TotalSegmentator的真实分割作为教师标签（而不是模型预测）
2. 增加HU值硬约束：空气/背景区域强制排除
3. 以灰度/HU值作为重点约束

核心思想：
- 从已标注的脊椎附近肌肉（autochthon, gluteus, iliopsoas）学习肌肉特征
- 让模型学会识别全身所有符合肌肉HU特征的区域
- 在已标注区域必须与标签一致
- 在未标注区域根据HU特征自行判断

作者: hzhang02
日期: 2026-01-10
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

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# 1. 常量定义
# ============================================================================

# 肌肉HU值范围 - 这是核心约束
MUSCLE_HU_MIN = -29
MUSCLE_HU_MAX = 150

# 硬排除区域 - 这些区域绝对不可能是肌肉
AIR_HU_MAX = -200      # 空气/背景，必须排除
BONE_HU_MIN = 300      # 骨骼，必须排除

# 身体区域阈值
BODY_HU_MIN = -500

# 已知肌肉分割文件（TotalSegmentator生成）
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

# 腹部区域标记器官（用于筛选腹部切片）
ABDOMINAL_ORGANS = [
    'liver.nii.gz',
    'spleen.nii.gz',
    'kidney_left.nii.gz',
    'kidney_right.nii.gz',
    'pancreas.nii.gz',
    'stomach.nii.gz',
    'colon.nii.gz',
    'small_bowel.nii.gz',
]

# 需要排除的区域标记（大腿）
EXCLUDE_MARKERS = [
    'femur_left.nii.gz',
    'femur_right.nii.gz',
]

# 是否只使用腹部切片
ABDOMINAL_ONLY = True

# 训练超参数
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
EPOCHS = 30

# 可视化配置
VIS_EVERY_N_EPOCHS = 2
VIS_NUM_SAMPLES = 4

# 损失权重 - 重点加强HU约束
LABEL_WEIGHT = 2.0            # 已标注区域一致性
HU_CONSTRAINT_WEIGHT = 3.0    # HU约束权重（加强！）
EXCLUSION_WEIGHT = 5.0        # 排除区域约束（最强！空气/骨骼必须排除）
BOUNDARY_WEIGHT = 0.3         # 边界平滑


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


class MuscleSegmentNet(nn.Module):
    """
    肌肉分割网络

    输入通道：
    - CT图像（归一化）
    - HU有效区域mask（在肌肉HU范围内）
    - HU排除区域mask（空气/骨骼，必须排除）
    """

    def __init__(self, in_ch=3, features=[32, 64, 128, 256]):
        super().__init__()

        # 编码器
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        current_ch = in_ch
        for f in features:
            self.encoder.append(DoubleConv(current_ch, f, dropout=0.1))
            current_ch = f

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=0.2)

        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        rev_features = list(reversed(features))
        up_ch = features[-1] * 2

        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(up_ch, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f, dropout=0.1))
            up_ch = f

        # 输出层
        self.final = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        # 编码
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码
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
# 3. 损失函数
# ============================================================================

class MuscleTransferLossV2(nn.Module):
    """
    肌肉迁移学习损失函数 V2

    重点改进：
    1. 硬排除约束：空气/骨骼区域预测必须为0
    2. HU约束加强：预测为肌肉的区域必须HU值合理
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, label_mask, hu_slice, body_mask, exclusion_mask):
        """
        Args:
            pred_logits: 模型预测 (B, 1, H, W)
            label_mask: 真实标签 (B, 1, H, W)，已知肌肉区域
            hu_slice: 原始HU值 (B, 1, H, W)
            body_mask: 身体掩码 (B, 1, H, W)
            exclusion_mask: 排除区域 (B, 1, H, W)，空气/骨骼
        """
        pred_prob = torch.sigmoid(pred_logits)
        losses = {}

        # 1. 标签一致性损失 - 在已标注区域必须一致
        has_label = (label_mask > 0.5).float()
        if has_label.sum() > 0:
            # 在有标签的区域计算BCE
            bce_loss = self.bce(pred_logits, label_mask)
            # 有标签区域权重更高
            weight = 1.0 + has_label * 2.0
            label_loss = (bce_loss * weight * body_mask).sum() / (body_mask.sum() + 1e-6)
        else:
            label_loss = torch.tensor(0.0, device=pred_logits.device)
        losses['label'] = label_loss * LABEL_WEIGHT

        # 2. 排除区域损失 - 空气/骨骼区域必须预测为0（最重要！）
        # 这是硬约束，权重最高
        if exclusion_mask.sum() > 0:
            # 在排除区域，预测概率应该为0
            exclusion_loss = (pred_prob * exclusion_mask).mean()
        else:
            exclusion_loss = torch.tensor(0.0, device=pred_logits.device)
        losses['exclusion'] = exclusion_loss * EXCLUSION_WEIGHT

        # 3. HU约束损失 - 预测为肌肉的区域HU值必须合理
        # 如果预测为肌肉但HU不在范围内，惩罚
        hu_valid = ((hu_slice >= MUSCLE_HU_MIN) & (hu_slice <= MUSCLE_HU_MAX)).float()
        hu_invalid = 1 - hu_valid

        # 在HU无效区域预测为肌肉的惩罚
        hu_violation = pred_prob * hu_invalid * body_mask * (1 - exclusion_mask)
        hu_loss = hu_violation.mean() * 10  # 放大惩罚
        losses['hu_constraint'] = hu_loss * HU_CONSTRAINT_WEIGHT

        # 4. 边界平滑损失
        tv_h = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]).mean()
        tv_w = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]).mean()
        losses['boundary'] = (tv_h + tv_w) * BOUNDARY_WEIGHT

        # 总损失
        total_loss = sum(losses.values())
        losses['total'] = total_loss

        return total_loss, losses


# ============================================================================
# 4. 数据集
# ============================================================================

class MuscleTransferDatasetV2(Dataset):
    """
    肌肉迁移学习数据集 V2

    直接使用TotalSegmentator的标签，不依赖模型预测
    支持腹部区域筛选
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

            # 检查是否有肌肉分割文件
            has_muscle = any(os.path.exists(os.path.join(seg_dir, f)) for f in self.muscle_files)
            if not has_muscle:
                continue

            img = nib.load(ct_path)
            depth = img.shape[2]

            # 如果需要腹部筛选，预加载标记器官
            if self.abdominal_only:
                # 加载腹部器官掩码
                abdominal_masks = []
                for organ in ABDOMINAL_ORGANS:
                    organ_path = os.path.join(seg_dir, organ)
                    if os.path.exists(organ_path):
                        abdominal_masks.append(nib.load(organ_path).get_fdata())

                # 加载排除标记（大腿骨）
                exclude_masks = []
                for marker in EXCLUDE_MARKERS:
                    marker_path = os.path.join(seg_dir, marker)
                    if os.path.exists(marker_path):
                        exclude_masks.append(nib.load(marker_path).get_fdata())

            for z in range(depth):
                total_slices += 1

                if self.abdominal_only:
                    # 检查是否是腹部切片：有腹部器官 AND 没有大腿骨
                    has_abdominal = any(
                        mask[:, :, z].sum() > 0 for mask in abdominal_masks
                    ) if abdominal_masks else False

                    has_femur = any(
                        mask[:, :, z].sum() > 0 for mask in exclude_masks
                    ) if exclude_masks else False

                    # 只保留腹部切片（有腹部器官或没有大腿骨的切片）
                    if has_femur:
                        continue  # 跳过有大腿骨的切片
                    if not has_abdominal and not abdominal_masks:
                        continue  # 如果没有腹部器官参考，跳过

                self.items.append((s, z))
                kept_slices += 1

        if self.abdominal_only:
            print(f"   腹部筛选: {kept_slices}/{total_slices} 切片保留 ({100*kept_slices/max(total_slices,1):.1f}%)")

    def __len__(self):
        return len(self.items)

    def _load_muscle_mask(self, subject_dir, z_idx, original_shape):
        """加载并合并所有已知肌肉的分割"""
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
        """获取身体掩码"""
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

        # 加载CT
        ct_path = os.path.join(subject_dir, 'ct.nii.gz')
        img = nib.load(ct_path).get_fdata().astype(np.float32)
        hu_slice = img[:, :, z]
        original_shape = hu_slice.shape[:2]

        # 加载真实肌肉标签（TotalSegmentator）
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

        # HU有效区域（在肌肉HU范围内）
        hu_valid = ((hu_resized >= MUSCLE_HU_MIN) & (hu_resized <= MUSCLE_HU_MAX)).astype(np.float32)

        # 排除区域（空气/骨骼，必须排除）
        exclusion = ((hu_resized < AIR_HU_MAX) | (hu_resized > BONE_HU_MIN)).astype(np.float32)

        # 身体掩码
        body_mask = self._get_body_mask(hu_resized)

        # 构建输入 (3通道: CT + HU有效区域 + 排除区域)
        input_tensor = np.stack([ct_resized, hu_valid, 1 - exclusion], axis=0)

        # 转换为tensor
        input_t = torch.from_numpy(input_tensor).float()
        hu_t = torch.from_numpy(hu_resized).unsqueeze(0).float()
        muscle_t = torch.from_numpy(muscle_resized).unsqueeze(0).float()
        body_t = torch.from_numpy(body_mask).unsqueeze(0).float()
        exclusion_t = torch.from_numpy(exclusion).unsqueeze(0).float()

        return input_t, hu_t, muscle_t, body_t, exclusion_t


# ============================================================================
# 5. 评估指标
# ============================================================================

def compute_metrics(pred, label, hu_slice, body_mask, exclusion_mask):
    """计算评估指标"""
    pred_binary = (pred > 0.5).float()
    label_binary = (label > 0.5).float()

    metrics = {}

    # 1. 标签区域覆盖率
    if label_binary.sum() > 0:
        coverage = (pred_binary * label_binary).sum() / label_binary.sum()
        metrics['label_coverage'] = coverage.item()
    else:
        metrics['label_coverage'] = 1.0

    # 2. 标签区域Dice
    if label_binary.sum() > 0:
        inter = (pred_binary * label_binary).sum()
        union = pred_binary.sum() + label_binary.sum()
        dice = (2 * inter) / (union + 1e-6)
        metrics['label_dice'] = dice.item()
    else:
        metrics['label_dice'] = 1.0

    # 3. HU合规率
    pred_muscle = pred_binary * body_mask
    if pred_muscle.sum() > 0:
        hu_valid = ((hu_slice >= MUSCLE_HU_MIN) & (hu_slice <= MUSCLE_HU_MAX)).float()
        compliance = (pred_muscle * hu_valid).sum() / pred_muscle.sum()
        metrics['hu_compliance'] = compliance.item()
    else:
        metrics['hu_compliance'] = 1.0

    # 4. 排除区域溢出率（关键指标！）
    if exclusion_mask.sum() > 0:
        overflow = (pred_binary * exclusion_mask).sum() / exclusion_mask.sum()
        metrics['exclusion_overflow'] = overflow.item()
    else:
        metrics['exclusion_overflow'] = 0.0

    # 5. 扩展比例
    if label_binary.sum() > 0:
        expansion = pred_muscle.sum() / label_binary.sum()
        metrics['expansion_ratio'] = expansion.item()
    else:
        metrics['expansion_ratio'] = 1.0

    return metrics


# ============================================================================
# 6. 可视化
# ============================================================================

def visualize_predictions(model, dataloader, device, epoch, output_dir, num_samples=4):
    """可视化预测结果"""
    model.eval()

    samples = []
    with torch.no_grad():
        for inputs, hu_slices, labels, body_masks, exclusion_masks in dataloader:
            if len(samples) >= num_samples:
                break

            inputs = inputs.to(device)
            pred_logits = model(inputs)
            pred_prob = torch.sigmoid(pred_logits)

            for i in range(min(inputs.size(0), num_samples - len(samples))):
                samples.append({
                    'ct': inputs[i, 0].cpu().numpy(),
                    'hu_valid': inputs[i, 1].cpu().numpy(),
                    'label': labels[i, 0].numpy(),
                    'pred': pred_prob[i, 0].cpu().numpy(),
                    'hu': hu_slices[i, 0].numpy(),
                    'exclusion': exclusion_masks[i, 0].numpy(),
                })
                if len(samples) >= num_samples:
                    break

    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample in enumerate(samples):
        ct = sample['ct']
        label = sample['label']
        pred = sample['pred']
        pred_binary = (pred > 0.5).astype(np.float32)
        hu = sample['hu']
        exclusion = sample['exclusion']

        # 1. CT原图
        axes[idx, 0].imshow(ct, cmap='gray')
        axes[idx, 0].set_title('CT Image')
        axes[idx, 0].axis('off')

        # 2. HU值图
        hu_display = np.clip(hu, -200, 300)
        im = axes[idx, 1].imshow(hu_display, cmap='jet', vmin=-200, vmax=300)
        axes[idx, 1].set_title(f'HU Values')
        axes[idx, 1].axis('off')

        # 3. 真实标签（TotalSegmentator）
        axes[idx, 2].imshow(ct, cmap='gray')
        axes[idx, 2].imshow(label, cmap='Reds', alpha=0.5)
        axes[idx, 2].set_title('Ground Truth (TotalSeg)')
        axes[idx, 2].axis('off')

        # 4. 模型预测
        axes[idx, 3].imshow(ct, cmap='gray')
        axes[idx, 3].imshow(pred, cmap='Blues', alpha=0.5)
        axes[idx, 3].set_title('Prediction')
        axes[idx, 3].axis('off')

        # 5. 排除区域检查
        # 红色：排除区域中被错误预测的
        # 绿色：正确预测的肌肉
        overlay = np.zeros((*ct.shape, 3))
        overlay[..., 0] = pred_binary * exclusion  # 红：排除区域溢出（错误）
        overlay[..., 1] = pred_binary * (1 - exclusion) * label  # 绿：正确覆盖
        overlay[..., 2] = pred_binary * (1 - exclusion) * (1 - label)  # 蓝：新发现
        axes[idx, 4].imshow(ct, cmap='gray')
        axes[idx, 4].imshow(overlay, alpha=0.6)
        axes[idx, 4].set_title('Analysis (R:Error, G:Correct, B:New)')
        axes[idx, 4].axis('off')

        # 6. 差异对比
        diff = np.zeros((*ct.shape, 3))
        diff[..., 0] = label * (1 - pred_binary)  # 红：漏检
        diff[..., 1] = pred_binary * (1 - label) * (1 - exclusion)  # 绿：新发现（合理）
        diff[..., 2] = label * pred_binary  # 蓝：正确匹配
        axes[idx, 5].imshow(ct, cmap='gray')
        axes[idx, 5].imshow(diff, alpha=0.6)
        axes[idx, 5].set_title('Diff (R:Miss, G:New, B:Match)')
        axes[idx, 5].axis('off')

    plt.suptitle(f'Epoch {epoch} - Segmentation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'visualization_epoch{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return wandb.Image(save_path, caption=f"Epoch {epoch} Predictions")


# ============================================================================
# 7. 主训练流程
# ============================================================================

def main():
    print("=" * 60)
    print("肌肉特征迁移学习 V2 - 基于TotalSegmentator标签")
    print("=" * 60)

    # 配置
    DATA_ROOT = '/local/hzhang02/data'
    OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs_muscle_transfer_v2'
    TARGET_SHAPE = (256, 256)
    NUM_SUBJECTS = 50

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== 初始化 wandb ==========
    wandb.init(
        project="muscle-transfer-learning",
        name=f"transfer-v2-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "version": "v2",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "num_subjects": NUM_SUBJECTS,
            "target_shape": TARGET_SHAPE,
            "label_weight": LABEL_WEIGHT,
            "hu_constraint_weight": HU_CONSTRAINT_WEIGHT,
            "exclusion_weight": EXCLUSION_WEIGHT,
            "boundary_weight": BOUNDARY_WEIGHT,
            "muscle_hu_min": MUSCLE_HU_MIN,
            "muscle_hu_max": MUSCLE_HU_MAX,
            "air_hu_max": AIR_HU_MAX,
            "bone_hu_min": BONE_HU_MIN,
            "known_muscles": KNOWN_MUSCLE_FILES,
            "abdominal_only": ABDOMINAL_ONLY,
            "abdominal_organs": ABDOMINAL_ORGANS,
            "exclude_markers": EXCLUDE_MARKERS,
        }
    )
    print("   wandb 初始化完成")

    # ========== 初始化模型 ==========
    print("\n1. 初始化模型...")
    model = MuscleSegmentNet(in_ch=3, features=[32, 64, 128, 256])
    model.to(device)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数
    criterion = MuscleTransferLossV2()

    # 优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ========== 加载数据 ==========
    print("\n2. 加载训练数据...")

    all_subjects_raw = [d for d in os.listdir(DATA_ROOT)
                        if d.startswith('s') and os.path.isdir(os.path.join(DATA_ROOT, d))]
    all_subjects = [s for s in all_subjects_raw if s in [f's{i:04d}' for i in range(NUM_SUBJECTS)]]
    subjects = [os.path.join(DATA_ROOT, s) for s in sorted(all_subjects)]

    print(f"   找到 {len(subjects)} 个受试者")

    # 划分训练/验证
    random.shuffle(subjects)
    n = len(subjects)
    ntrain = max(1, int(n * 0.8))
    train_subs = subjects[:ntrain]
    val_subs = subjects[ntrain:]

    print(f"   训练集: {len(train_subs)} / 验证集: {len(val_subs)}")

    # 创建数据集（启用腹部筛选）
    train_ds = MuscleTransferDatasetV2(train_subs, TARGET_SHAPE, abdominal_only=ABDOMINAL_ONLY)
    val_ds = MuscleTransferDatasetV2(val_subs, TARGET_SHAPE, abdominal_only=ABDOMINAL_ONLY)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"   训练切片: {len(train_ds)} / 验证切片: {len(val_ds)}")

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ========== 训练循环 ==========
    print("\n3. 开始训练...")
    print("=" * 60)

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'label_coverage': [], 'label_dice': [],
        'hu_compliance': [], 'exclusion_overflow': [], 'expansion_ratio': []
    }

    best_dice = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)

        # 训练阶段
        model.train()
        train_losses = []
        train_loss_components = {'label': [], 'hu_constraint': [], 'exclusion': [], 'boundary': []}

        for inputs, hu_slices, labels, body_masks, exclusion_masks in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            hu_slices = hu_slices.to(device)
            labels = labels.to(device)
            body_masks = body_masks.to(device)
            exclusion_masks = exclusion_masks.to(device)

            optimizer.zero_grad()

            pred_logits = model(inputs)
            loss, loss_dict = criterion(pred_logits, labels, hu_slices, body_masks, exclusion_masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            for k in train_loss_components:
                if k in loss_dict:
                    train_loss_components[k].append(loss_dict[k].item())

        # 验证阶段
        model.eval()
        val_losses = []
        val_metrics = {'label_coverage': [], 'label_dice': [],
                       'hu_compliance': [], 'exclusion_overflow': [], 'expansion_ratio': []}

        with torch.no_grad():
            for inputs, hu_slices, labels, body_masks, exclusion_masks in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                hu_slices = hu_slices.to(device)
                labels = labels.to(device)
                body_masks = body_masks.to(device)
                exclusion_masks = exclusion_masks.to(device)

                pred_logits = model(inputs)
                loss, _ = criterion(pred_logits, labels, hu_slices, body_masks, exclusion_masks)
                val_losses.append(loss.item())

                pred_prob = torch.sigmoid(pred_logits)
                for i in range(inputs.size(0)):
                    metrics = compute_metrics(
                        pred_prob[i], labels[i], hu_slices[i],
                        body_masks[i], exclusion_masks[i]
                    )
                    for k, v in metrics.items():
                        val_metrics[k].append(v)

        scheduler.step()

        # 记录结果
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
        print(f"  标签覆盖率: {avg_metrics['label_coverage']:.4f}")
        print(f"  标签Dice: {avg_metrics['label_dice']:.4f}")
        print(f"  HU合规率: {avg_metrics['hu_compliance']:.4f}")
        print(f"  排除区域溢出: {avg_metrics['exclusion_overflow']:.4f} (越低越好)")
        print(f"  扩展比例: {avg_metrics['expansion_ratio']:.2f}x")
        print(f"  耗时: {epoch_time:.1f}秒")

        # wandb记录
        log_dict = {
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/label_loss": np.mean(train_loss_components['label']),
            "train/hu_constraint_loss": np.mean(train_loss_components['hu_constraint']),
            "train/exclusion_loss": np.mean(train_loss_components['exclusion']),
            "train/boundary_loss": np.mean(train_loss_components['boundary']),
            "val/loss": avg_val_loss,
            "val/label_coverage": avg_metrics['label_coverage'],
            "val/label_dice": avg_metrics['label_dice'],
            "val/hu_compliance": avg_metrics['hu_compliance'],
            "val/exclusion_overflow": avg_metrics['exclusion_overflow'],
            "val/expansion_ratio": avg_metrics['expansion_ratio'],
            "lr": scheduler.get_last_lr()[0],
        }

        # 定期可视化
        if epoch % VIS_EVERY_N_EPOCHS == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  生成可视化...")
            vis_image = visualize_predictions(model, val_loader, device, epoch, OUTPUT_DIR, VIS_NUM_SAMPLES)
            log_dict["predictions"] = vis_image

        wandb.log(log_dict)

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'metrics': avg_metrics,
        }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch{epoch}.pth'))

        # 保存最佳模型
        if avg_metrics['label_dice'] > best_dice:
            best_dice = avg_metrics['label_dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': avg_metrics,
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  ★ 新最佳模型！Dice: {best_dice:.4f}")

    # ========== 保存结果 ==========
    print("\n4. 保存训练结果...")

    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 可视化训练曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['epoch'], history['label_dice'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].set_title('Label Dice (↑)')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['epoch'], history['hu_compliance'])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Compliance')
    axes[0, 2].set_title('HU Compliance (↑)')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history['epoch'], history['exclusion_overflow'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Overflow')
    axes[1, 0].set_title('Exclusion Overflow (↓)')
    axes[1, 0].set_ylim([0, 0.5])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['epoch'], history['label_coverage'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Coverage')
    axes[1, 1].set_title('Label Coverage (↑)')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history['epoch'], history['expansion_ratio'])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Ratio')
    axes[1, 2].set_title('Expansion Ratio')
    axes[1, 2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
    plt.close()

    wandb.log({"training_curves": wandb.Image(os.path.join(OUTPUT_DIR, 'training_history.png'))})
    wandb.summary["best_dice"] = best_dice
    wandb.finish()

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  最佳Dice: {best_dice:.4f}")
    print(f"  结果保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
