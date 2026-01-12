#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肌肉特征迁移学习 - 全身肌肉分割

核心思想：
1. 从已标注的脊椎附近肌肉学习"肌肉的特征"（HU值分布、纹理等）
2. 让模型对整个CT图像进行全肌肉分割
3. 在已标注区域，要求新模型与Teacher模型高度一致（监督信号）
4. 在未标注区域（如腹部），模型根据学到的肌肉特征自行分割

损失函数设计：
- L_consistency: 在Teacher预测区域，新模型必须与Teacher一致
- L_hu_prior: 预测为肌肉的区域，HU值应该在肌肉范围内
- L_boundary: 边界平滑性约束
- L_morphology: 形态学合理性约束

作者: hzhang02
日期: 2026-01-10
"""

import os
import json
import random
import time
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import sobel
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

# 肌肉HU值范围
MUSCLE_HU_MIN = -29
MUSCLE_HU_MAX = 150
MUSCLE_HU_OPTIMAL_MIN = 0  # 典型肌肉HU值
MUSCLE_HU_OPTIMAL_MAX = 100

# 其他组织HU值范围（用于负样本）
BONE_HU_MIN = 200  # 骨骼
FAT_HU_MIN = -190  # 脂肪
FAT_HU_MAX = -30
AIR_HU_MAX = -500  # 空气

# 训练超参数
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
EPOCHS = 30

# 可视化配置
VIS_EVERY_N_EPOCHS = 2  # 每N个epoch可视化一次
VIS_NUM_SAMPLES = 4     # 每次可视化的样本数

# 损失权重
CONSISTENCY_WEIGHT = 2.0      # Teacher一致性损失权重
HU_PRIOR_WEIGHT = 1.0         # HU先验损失权重
BOUNDARY_WEIGHT = 0.5         # 边界平滑损失权重
COVERAGE_WEIGHT = 1.0         # 覆盖率损失权重（不能丢失已知肌肉）


# ============================================================================
# 2. 网络结构定义
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """注意力模块 - 帮助模型关注肌肉特征"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 调整大小以匹配
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MuscleTransferNet(nn.Module):
    """
    肌肉特征迁移网络

    输入:
        - CT图像 (1通道)
        - HU特征图 (1通道，表示是否在肌肉HU范围内)
        - Teacher预测 (1通道，已知肌肉区域)
    输出:
        - 全身肌肉分割 (1通道，sigmoid后为概率)
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

        # 解码器（带注意力）
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()

        rev_features = list(reversed(features))
        up_ch = features[-1] * 2

        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(up_ch, f, kernel_size=2, stride=2))
            self.attention.append(AttentionBlock(F_g=f, F_l=f, F_int=f // 2))
            self.decoder.append(DoubleConv(f * 2, f, dropout=0.1))
            up_ch = f

        # 输出层
        self.final = nn.Conv2d(features[0], 1, kernel_size=1)

        # 肌肉特征编码器（学习肌肉的通用特征）
        self.muscle_feature_encoder = nn.Sequential(
            nn.Conv2d(features[-1] * 2, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        # 编码
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 提取肌肉特征向量（可用于后续分析）
        muscle_features = self.muscle_feature_encoder(x)

        # 解码（带注意力）
        skips = skips[::-1]
        for i, (upconv, attn, dec) in enumerate(zip(self.upconvs, self.attention, self.decoder)):
            x = upconv(x)
            skip = skips[i]

            # 调整大小
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            # 注意力
            skip = attn(x, skip)

            # 拼接和卷积
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        # 输出
        out = self.final(x)

        return out, muscle_features


class DoubleConvTeacher(nn.Module):
    """Teacher模型的双卷积块（无dropout，与原训练一致）"""
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
    """Teacher模型结构（与原训练一致）"""
    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for f in features:
            self.downs.append(DoubleConvTeacher(in_ch, f))
            in_ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvTeacher(features[-1], features[-1] * 2)

        rev = list(reversed(features))
        up_in = features[-1] * 2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConvTeacher(up_in, f))
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
# 3. 损失函数
# ============================================================================

class MuscleTransferLoss(nn.Module):
    """
    肌肉迁移学习损失函数

    组成部分：
    1. 一致性损失：在Teacher预测区域，新模型必须与Teacher一致
    2. HU先验损失：预测为肌肉的区域，HU值应该在肌肉范围内
    3. 覆盖率损失：不能丢失Teacher预测的肌肉区域
    4. 边界损失：预测边界应该与HU值边界对齐
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, teacher_mask, hu_slice, body_mask):
        """
        Args:
            pred_logits: 模型预测 (B, 1, H, W)，未经sigmoid
            teacher_mask: Teacher模型预测 (B, 1, H, W)，已知肌肉区域
            hu_slice: 原始HU值 (B, 1, H, W)
            body_mask: 身体掩码 (B, 1, H, W)
        """
        pred_prob = torch.sigmoid(pred_logits)
        losses = {}

        # 1. 一致性损失 - 在Teacher预测区域，必须与Teacher一致
        # 使用加权BCE，Teacher区域权重更高
        teacher_region = (teacher_mask > 0.5).float()
        weight_map = torch.ones_like(teacher_mask)
        weight_map = weight_map + teacher_region * 2.0  # Teacher区域权重更高

        bce_loss = self.bce(pred_logits, teacher_mask)
        consistency_loss = (bce_loss * weight_map * body_mask).sum() / (body_mask.sum() + 1e-6)
        losses['consistency'] = consistency_loss * CONSISTENCY_WEIGHT

        # 2. HU先验损失 - 预测为肌肉的区域，HU值应该在肌肉范围内
        hu_in_range = ((hu_slice >= MUSCLE_HU_MIN) & (hu_slice <= MUSCLE_HU_MAX)).float()
        hu_optimal = ((hu_slice >= MUSCLE_HU_OPTIMAL_MIN) & (hu_slice <= MUSCLE_HU_OPTIMAL_MAX)).float()

        # 如果预测为肌肉但HU不在范围内，惩罚
        hu_violation = pred_prob * (1 - hu_in_range) * body_mask
        hu_prior_loss = hu_violation.mean()
        losses['hu_prior'] = hu_prior_loss * HU_PRIOR_WEIGHT

        # 3. 覆盖率损失 - 不能丢失Teacher预测的肌肉
        # 在Teacher预测为肌肉的区域，新模型也必须预测为肌肉
        if teacher_region.sum() > 0:
            coverage = (pred_prob * teacher_region).sum() / (teacher_region.sum() + 1e-6)
            coverage_loss = 1 - coverage  # 覆盖率越高，损失越低
        else:
            coverage_loss = torch.tensor(0.0, device=pred_logits.device)
        losses['coverage'] = coverage_loss * COVERAGE_WEIGHT

        # 4. 边界平滑损失 - 预测应该有平滑的边界
        # 使用总变差（TV）损失
        tv_h = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]).mean()
        tv_w = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]).mean()
        boundary_loss = tv_h + tv_w
        losses['boundary'] = boundary_loss * BOUNDARY_WEIGHT

        # 总损失
        total_loss = sum(losses.values())
        losses['total'] = total_loss

        return total_loss, losses


# ============================================================================
# 4. 数据集
# ============================================================================

class MuscleTransferDataset(Dataset):
    """
    肌肉迁移学习数据集

    为每个切片提供：
    - CT图像（归一化）
    - 原始HU值
    - HU特征图（是否在肌肉范围内）
    - Teacher预测（已知肌肉区域）
    - 身体掩码
    """

    def __init__(self, subjects, teacher_model, device, target_shape=(256, 256)):
        self.items = []
        self.target_shape = target_shape
        self.teacher_model = teacher_model
        self.device = device

        print("构建数据集...")
        for s in tqdm(subjects, desc="扫描受试者"):
            ct_path = os.path.join(s, 'ct.nii.gz')
            if not os.path.exists(ct_path):
                continue

            img = nib.load(ct_path)
            depth = img.shape[2]

            for z in range(depth):
                self.items.append((ct_path, z))

    def __len__(self):
        return len(self.items)

    def _get_body_mask(self, hu_slice):
        """获取身体掩码"""
        body = hu_slice > -500
        body = morphology.binary_closing(body, morphology.disk(3))
        body = morphology.binary_opening(body, morphology.disk(2))
        body = ndimage.binary_fill_holes(body)
        labeled = label(body)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            largest = max(regions, key=lambda x: x.area)
            body = labeled == largest.label
        return body.astype(np.float32)

    def _get_teacher_prediction(self, ct_normalized, original_shape):
        """获取Teacher模型预测"""
        H, W = self.target_shape
        ct_resized = resize(ct_normalized, (H, W), order=1, preserve_range=True)

        img_t = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            pred = torch.sigmoid(self.teacher_model(img_t))
            # 合并所有肌肉类别
            pred_combined = (pred[0].sum(dim=0) > 0.5).float().cpu().numpy()

        return pred_combined

    def __getitem__(self, idx):
        ct_path, z = self.items[idx]

        # 加载CT
        img = nib.load(ct_path).get_fdata().astype(np.float32)
        hu_slice = img[:, :, z]
        original_shape = hu_slice.shape

        # 归一化CT
        lo, hi = np.percentile(hu_slice, 1), np.percentile(hu_slice, 99)
        if hi - lo > 0:
            ct_normalized = np.clip(hu_slice, lo, hi)
            ct_normalized = (ct_normalized - lo) / (hi - lo)
        else:
            ct_normalized = np.zeros_like(hu_slice)

        # 获取Teacher预测
        teacher_pred = self._get_teacher_prediction(ct_normalized, original_shape)

        # 调整所有内容到目标大小
        H, W = self.target_shape
        ct_resized = resize(ct_normalized, (H, W), order=1, preserve_range=True)
        hu_resized = resize(hu_slice, (H, W), order=1, preserve_range=True)

        # HU特征图：是否在肌肉HU范围内
        hu_muscle_feature = ((hu_resized >= MUSCLE_HU_MIN) & (hu_resized <= MUSCLE_HU_MAX)).astype(np.float32)

        # 身体掩码
        body_mask = self._get_body_mask(hu_resized)

        # 构建输入 (3通道: CT + HU特征 + Teacher预测)
        input_tensor = np.stack([ct_resized, hu_muscle_feature, teacher_pred], axis=0)

        # 转换为tensor
        input_t = torch.from_numpy(input_tensor).float()
        hu_t = torch.from_numpy(hu_resized).unsqueeze(0).float()
        teacher_t = torch.from_numpy(teacher_pred).unsqueeze(0).float()
        body_t = torch.from_numpy(body_mask).unsqueeze(0).float()

        return input_t, hu_t, teacher_t, body_t


# ============================================================================
# 5. 评估指标
# ============================================================================

def compute_metrics(pred, teacher, hu_slice, body_mask):
    """
    计算评估指标

    Returns:
        dict: 包含各种评估指标
    """
    pred_binary = (pred > 0.5).float()
    teacher_binary = (teacher > 0.5).float()

    metrics = {}

    # 1. Teacher区域覆盖率（新模型是否覆盖了已知肌肉）
    if teacher_binary.sum() > 0:
        coverage = (pred_binary * teacher_binary).sum() / teacher_binary.sum()
        metrics['teacher_coverage'] = coverage.item()
    else:
        metrics['teacher_coverage'] = 1.0

    # 2. Teacher区域Dice
    if teacher_binary.sum() > 0:
        inter = (pred_binary * teacher_binary).sum()
        union = pred_binary.sum() + teacher_binary.sum()
        dice = (2 * inter) / (union + 1e-6)
        metrics['teacher_dice'] = dice.item()
    else:
        metrics['teacher_dice'] = 1.0

    # 3. HU合规率（预测为肌肉的区域中，HU值在范围内的比例）
    pred_muscle = pred_binary * body_mask
    if pred_muscle.sum() > 0:
        hu_in_range = ((hu_slice >= MUSCLE_HU_MIN) & (hu_slice <= MUSCLE_HU_MAX)).float()
        compliance = (pred_muscle * hu_in_range).sum() / pred_muscle.sum()
        metrics['hu_compliance'] = compliance.item()
    else:
        metrics['hu_compliance'] = 1.0

    # 4. 扩展比例（新预测相比Teacher的扩展程度）
    if teacher_binary.sum() > 0:
        expansion = pred_muscle.sum() / teacher_binary.sum()
        metrics['expansion_ratio'] = expansion.item()
    else:
        metrics['expansion_ratio'] = 1.0

    # 5. 溢出率（预测到非肌肉HU区域的比例）
    if pred_muscle.sum() > 0:
        hu_out_range = ((hu_slice < MUSCLE_HU_MIN) | (hu_slice > MUSCLE_HU_MAX)).float()
        overflow = (pred_muscle * hu_out_range).sum() / pred_muscle.sum()
        metrics['overflow_rate'] = overflow.item()
    else:
        metrics['overflow_rate'] = 0.0

    return metrics


# ============================================================================
# 6. 可视化函数
# ============================================================================

def visualize_predictions(model, dataloader, device, epoch, output_dir, num_samples=4):
    """
    可视化模型预测结果

    生成对比图：CT原图 | Teacher预测 | Student预测 | 差异图

    Args:
        model: Student模型
        dataloader: 数据加载器
        device: 设备
        epoch: 当前epoch
        output_dir: 输出目录
        num_samples: 可视化样本数量

    Returns:
        wandb.Image: 可上传到wandb的图像对象
    """
    model.eval()

    # 收集样本
    samples = []
    with torch.no_grad():
        for inputs, hu_slices, teacher_masks, body_masks in dataloader:
            if len(samples) >= num_samples:
                break

            inputs = inputs.to(device)
            pred_logits, _ = model(inputs)
            pred_prob = torch.sigmoid(pred_logits)

            for i in range(min(inputs.size(0), num_samples - len(samples))):
                samples.append({
                    'ct': inputs[i, 0].cpu().numpy(),  # CT图像
                    'hu_feature': inputs[i, 1].cpu().numpy(),  # HU特征图
                    'teacher': teacher_masks[i, 0].numpy(),  # Teacher预测
                    'student': pred_prob[i, 0].cpu().numpy(),  # Student预测
                    'hu': hu_slices[i, 0].numpy(),  # 原始HU值
                    'body': body_masks[i, 0].numpy(),  # 身体掩码
                })

                if len(samples) >= num_samples:
                    break

    # 创建可视化图
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample in enumerate(samples):
        ct = sample['ct']
        teacher = sample['teacher']
        student = sample['student']
        student_binary = (student > 0.5).astype(np.float32)
        hu = sample['hu']

        # 计算差异：Student预测但Teacher没预测的区域（扩展区域）
        expansion = student_binary * (1 - teacher)
        # Teacher预测但Student没预测的区域（丢失区域）
        missing = teacher * (1 - student_binary)

        # 1. CT原图
        axes[idx, 0].imshow(ct, cmap='gray')
        axes[idx, 0].set_title('CT Image')
        axes[idx, 0].axis('off')

        # 2. Teacher预测（已知肌肉）
        axes[idx, 1].imshow(ct, cmap='gray')
        axes[idx, 1].imshow(teacher, cmap='Reds', alpha=0.5)
        axes[idx, 1].set_title('Teacher (Known Muscle)')
        axes[idx, 1].axis('off')

        # 3. Student预测（全肌肉）
        axes[idx, 2].imshow(ct, cmap='gray')
        axes[idx, 2].imshow(student, cmap='Blues', alpha=0.5)
        axes[idx, 2].set_title(f'Student (Predicted)')
        axes[idx, 2].axis('off')

        # 4. 对比图：Teacher(红) vs Student(蓝) 重叠(紫)
        overlay = np.zeros((*ct.shape, 3))
        overlay[..., 0] = teacher * 0.7  # 红色：Teacher
        overlay[..., 2] = student_binary * 0.7  # 蓝色：Student
        # 重叠区域变成紫色
        axes[idx, 3].imshow(ct, cmap='gray')
        axes[idx, 3].imshow(overlay, alpha=0.6)
        axes[idx, 3].set_title('Overlay (Red:Teacher, Blue:Student)')
        axes[idx, 3].axis('off')

        # 5. 差异分析：扩展(绿) vs 丢失(红)
        diff_overlay = np.zeros((*ct.shape, 3))
        diff_overlay[..., 1] = expansion * 0.8  # 绿色：扩展区域
        diff_overlay[..., 0] = missing * 0.8  # 红色：丢失区域
        axes[idx, 4].imshow(ct, cmap='gray')
        axes[idx, 4].imshow(diff_overlay, alpha=0.7)
        axes[idx, 4].set_title('Diff (Green:New, Red:Missing)')
        axes[idx, 4].axis('off')

    plt.suptitle(f'Epoch {epoch} - Segmentation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, f'visualization_epoch{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return wandb.Image(save_path, caption=f"Epoch {epoch} Predictions")


def visualize_single_prediction(ct, teacher, student, hu, save_path=None):
    """
    单张图像的详细可视化

    Args:
        ct: CT图像 (H, W)
        teacher: Teacher预测 (H, W)
        student: Student预测概率 (H, W)
        hu: 原始HU值 (H, W)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    student_binary = (student > 0.5).astype(np.float32)

    # 1. CT原图
    axes[0, 0].imshow(ct, cmap='gray')
    axes[0, 0].set_title('CT Image')
    axes[0, 0].axis('off')

    # 2. HU值图（肌肉范围高亮）
    hu_display = np.clip(hu, -200, 300)
    axes[0, 1].imshow(hu_display, cmap='jet')
    axes[0, 1].set_title(f'HU Values (Muscle: {MUSCLE_HU_MIN}~{MUSCLE_HU_MAX})')
    axes[0, 1].axis('off')
    # 添加colorbar
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)

    # 3. Teacher预测
    axes[0, 2].imshow(ct, cmap='gray')
    axes[0, 2].imshow(teacher, cmap='Reds', alpha=0.5)
    axes[0, 2].set_title('Teacher Prediction (Known Muscle)')
    axes[0, 2].axis('off')

    # 4. Student预测概率
    axes[1, 0].imshow(student, cmap='hot')
    axes[1, 0].set_title('Student Probability Map')
    axes[1, 0].axis('off')
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046)

    # 5. Student二值化预测
    axes[1, 1].imshow(ct, cmap='gray')
    axes[1, 1].imshow(student_binary, cmap='Blues', alpha=0.5)
    axes[1, 1].set_title('Student Prediction (Threshold=0.5)')
    axes[1, 1].axis('off')

    # 6. 对比分析
    expansion = student_binary * (1 - teacher)
    missing = teacher * (1 - student_binary)
    overlap = student_binary * teacher

    comparison = np.zeros((*ct.shape, 3))
    comparison[..., 0] = missing  # 红：丢失
    comparison[..., 1] = expansion  # 绿：新增
    comparison[..., 2] = overlap  # 蓝：重叠

    axes[1, 2].imshow(ct, cmap='gray')
    axes[1, 2].imshow(comparison, alpha=0.6)
    axes[1, 2].set_title('Analysis (R:Missing, G:New, B:Overlap)')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None


# ============================================================================
# 7. 主训练流程
# ============================================================================

def main():
    print("=" * 60)
    print("肌肉特征迁移学习 - 全身肌肉分割")
    print("=" * 60)

    # 配置
    DATA_ROOT = '/local/hzhang02/data'
    OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs_muscle_transfer'
    TEACHER_MODEL_PATH = '/local/hzhang02/data/dataset/outputs/best_model.pth'
    LABEL_MAP_PATH = '/local/hzhang02/data/dataset/outputs/label_map.json'

    TARGET_SHAPE = (256, 256)
    NUM_SUBJECTS = 50

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== 初始化 wandb ==========
    wandb.init(
        project="muscle-transfer-learning",
        name=f"transfer-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "num_subjects": NUM_SUBJECTS,
            "target_shape": TARGET_SHAPE,
            "consistency_weight": CONSISTENCY_WEIGHT,
            "hu_prior_weight": HU_PRIOR_WEIGHT,
            "boundary_weight": BOUNDARY_WEIGHT,
            "coverage_weight": COVERAGE_WEIGHT,
            "muscle_hu_min": MUSCLE_HU_MIN,
            "muscle_hu_max": MUSCLE_HU_MAX,
            "vis_every_n_epochs": VIS_EVERY_N_EPOCHS,
            "vis_num_samples": VIS_NUM_SAMPLES,
        }
    )
    print("   wandb 初始化完成")

    # ========== 加载Teacher模型 ==========
    print("\n1. 加载Teacher模型...")

    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"   Teacher模型分割 {num_classes} 种肌肉: {list(label_map.keys())}")

    teacher_model = UNet2D(in_ch=1, out_ch=num_classes, features=[32, 64, 128, 256])
    checkpoint = torch.load(TEACHER_MODEL_PATH, map_location=device)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.to(device)
    teacher_model.eval()
    print(f"   Teacher模型加载成功")
    print(f"   Teacher验证Dice: {checkpoint.get('val_dice', 'N/A'):.4f}")

    # ========== 初始化Student模型 ==========
    print("\n2. 初始化Student模型...")

    student_model = MuscleTransferNet(in_ch=3, features=[32, 64, 128, 256])
    student_model.to(device)
    print(f"   Student模型参数量: {sum(p.numel() for p in student_model.parameters()):,}")

    # 损失函数
    criterion = MuscleTransferLoss()

    # 优化器
    optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ========== 加载数据 ==========
    print("\n3. 加载训练数据...")

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

    # 创建数据集
    train_ds = MuscleTransferDataset(train_subs, teacher_model, device, TARGET_SHAPE)
    val_ds = MuscleTransferDataset(val_subs, teacher_model, device, TARGET_SHAPE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)  # num_workers=0 因为dataset中使用了CUDA
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)  # num_workers=0 因为dataset中使用了CUDA

    print(f"   训练切片: {len(train_ds)} / 验证切片: {len(val_ds)}")

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ========== 训练循环 ==========
    print("\n4. 开始训练...")
    print("=" * 60)

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'teacher_coverage': [], 'teacher_dice': [],
        'hu_compliance': [], 'expansion_ratio': [], 'overflow_rate': []
    }

    best_teacher_dice = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)

        # 训练阶段
        student_model.train()
        train_losses = []
        train_loss_components = {'consistency': [], 'hu_prior': [], 'coverage': [], 'boundary': []}

        for inputs, hu_slices, teacher_masks, body_masks in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            hu_slices = hu_slices.to(device)
            teacher_masks = teacher_masks.to(device)
            body_masks = body_masks.to(device)

            optimizer.zero_grad()

            pred_logits, _ = student_model(inputs)
            loss, loss_dict = criterion(pred_logits, teacher_masks, hu_slices, body_masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            for k in train_loss_components:
                if k in loss_dict:
                    train_loss_components[k].append(loss_dict[k].item())

        # 验证阶段
        student_model.eval()
        val_losses = []
        val_metrics = {'teacher_coverage': [], 'teacher_dice': [],
                       'hu_compliance': [], 'expansion_ratio': [], 'overflow_rate': []}

        with torch.no_grad():
            for inputs, hu_slices, teacher_masks, body_masks in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                hu_slices = hu_slices.to(device)
                teacher_masks = teacher_masks.to(device)
                body_masks = body_masks.to(device)

                pred_logits, _ = student_model(inputs)
                loss, _ = criterion(pred_logits, teacher_masks, hu_slices, body_masks)
                val_losses.append(loss.item())

                # 计算评估指标
                pred_prob = torch.sigmoid(pred_logits)
                for i in range(inputs.size(0)):
                    metrics = compute_metrics(
                        pred_prob[i], teacher_masks[i],
                        hu_slices[i], body_masks[i]
                    )
                    for k, v in metrics.items():
                        val_metrics[k].append(v)

        # 更新学习率
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
        print(f"  Teacher覆盖率: {avg_metrics['teacher_coverage']:.4f}")
        print(f"  Teacher Dice: {avg_metrics['teacher_dice']:.4f}")
        print(f"  HU合规率: {avg_metrics['hu_compliance']:.4f}")
        print(f"  扩展比例: {avg_metrics['expansion_ratio']:.2f}x")
        print(f"  溢出率: {avg_metrics['overflow_rate']:.4f}")
        print(f"  耗时: {epoch_time:.1f}秒")

        # 记录到wandb
        log_dict = {
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/consistency_loss": np.mean(train_loss_components['consistency']),
            "train/hu_prior_loss": np.mean(train_loss_components['hu_prior']),
            "train/coverage_loss": np.mean(train_loss_components['coverage']),
            "train/boundary_loss": np.mean(train_loss_components['boundary']),
            "val/loss": avg_val_loss,
            "val/teacher_coverage": avg_metrics['teacher_coverage'],
            "val/teacher_dice": avg_metrics['teacher_dice'],
            "val/hu_compliance": avg_metrics['hu_compliance'],
            "val/expansion_ratio": avg_metrics['expansion_ratio'],
            "val/overflow_rate": avg_metrics['overflow_rate'],
            "lr": scheduler.get_last_lr()[0],
        }

        # 定期可视化预测结果
        if epoch % VIS_EVERY_N_EPOCHS == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  生成可视化...")
            vis_image = visualize_predictions(
                student_model, val_loader, device, epoch,
                OUTPUT_DIR, num_samples=VIS_NUM_SAMPLES
            )
            log_dict["predictions"] = vis_image
            print(f"  可视化已保存: visualization_epoch{epoch}.png")

        wandb.log(log_dict)

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'metrics': avg_metrics,
        }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch{epoch}.pth'))

        # 保存最佳模型（基于Teacher Dice）
        if avg_metrics['teacher_dice'] > best_teacher_dice:
            best_teacher_dice = avg_metrics['teacher_dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'metrics': avg_metrics,
            }, os.path.join(OUTPUT_DIR, 'best_transfer_model.pth'))
            print(f"  ★ 新最佳模型！Teacher Dice: {best_teacher_dice:.4f}")
            wandb.save(os.path.join(OUTPUT_DIR, 'best_transfer_model.pth'))

    # ========== 保存结果 ==========
    print("\n5. 保存训练结果...")

    # 保存历史
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 可视化训练曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training/Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['epoch'], history['teacher_coverage'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].set_title('Teacher Coverage (↑ better)')
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['epoch'], history['teacher_dice'])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice')
    axes[0, 2].set_title('Teacher Region Dice (↑ better)')
    axes[0, 2].set_ylim([0, 1.1])
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history['epoch'], history['hu_compliance'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Compliance')
    axes[1, 0].set_title('HU Compliance (↑ better)')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['epoch'], history['expansion_ratio'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_title('Expansion Ratio')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history['epoch'], history['overflow_rate'])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Rate')
    axes[1, 2].set_title('Overflow Rate (↓ better)')
    axes[1, 2].set_ylim([0, 0.5])
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
    plt.close()

    wandb.log({"training_curves": wandb.Image(os.path.join(OUTPUT_DIR, 'training_history.png'))})

    # 结束wandb
    wandb.summary["best_teacher_dice"] = best_teacher_dice
    wandb.finish()

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  最佳Teacher Dice: {best_teacher_dice:.4f}")
    print(f"  结果保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
