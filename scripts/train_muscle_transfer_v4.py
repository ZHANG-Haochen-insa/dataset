#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肌肉特征迁移学习 V4 - 注意力机制版本

核心思想：
1. 空间自注意力：学习图像不同位置之间的相似性
2. 跨区域注意力：以已标注肌肉区域为参考，关注未标注但特征相似的区域
3. 位置编码：让网络感知空间位置信息

网络会学习到：某个位置的特征如果和已标注肌肉区域的特征相似，
那么它很可能也是肌肉区域。

作者: hzhang02
日期: 2026-01-12
"""

import os
import json
import random
import time
import math
from typing import List, Dict, Optional
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
BATCH_SIZE = 8  # 注意力机制显存占用较大，减小batch size
EPOCHS = 30

# 可视化
VIS_EVERY_N_EPOCHS = 2
VIS_NUM_SAMPLES = 4

# 损失权重
LABEL_ALIGNMENT_WEIGHT = 3.0
COVERAGE_REWARD_WEIGHT = 1.0
EXCLUSION_WEIGHT = 5.0
SMOOTHNESS_WEIGHT = 0.2
ATTENTION_SIMILARITY_WEIGHT = 1.5  # 注意力相似性损失权重

# 是否只用腹部
ABDOMINAL_ONLY = True


# ============================================================================
# 2. 注意力模块
# ============================================================================

class PositionalEncoding2D(nn.Module):
    """2D正弦位置编码"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成位置编码
        y_pos = torch.arange(H, device=x.device).float().unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, device=x.device).float().unsqueeze(0).repeat(H, 1)

        # 归一化到 [0, 1]
        y_pos = y_pos / max(H - 1, 1)
        x_pos = x_pos / max(W - 1, 1)

        # 正弦编码
        channels_per_dim = self.channels // 4
        div_term = torch.exp(torch.arange(0, channels_per_dim, device=x.device).float() *
                            (-math.log(10000.0) / channels_per_dim))

        pe = torch.zeros(self.channels, H, W, device=x.device)

        # Y方向编码
        pe[0:channels_per_dim, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1) * math.pi)
        pe[channels_per_dim:2*channels_per_dim, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1) * math.pi)

        # X方向编码
        pe[2*channels_per_dim:3*channels_per_dim, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1) * math.pi)
        pe[3*channels_per_dim:4*channels_per_dim, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1) * math.pi)

        # 扩展batch维度
        pe = pe.unsqueeze(0).repeat(B, 1, 1, 1)

        return x + pe


class MultiHeadSelfAttention2D(nn.Module):
    """
    2D多头自注意力模块

    让特征图中的每个位置都能关注其他所有位置
    """

    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

        # 层归一化
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 残差连接
        identity = x
        x = self.norm(x)

        # 生成Q, K, V
        qkv = self.qkv(x)  # (B, 3C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(out)

        return identity + out


class CrossRegionAttention(nn.Module):
    """
    跨区域注意力模块

    核心思想：以已标注肌肉区域为参考（Query来源），
    去关注所有位置（Key/Value来源），找到特征相似的区域

    这样网络就能学习到"什么样的特征代表肌肉"
    """

    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # 分开的Q和KV投影
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.kv_proj = nn.Conv2d(channels, channels * 2, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.GroupNorm(8, channels)
        self.norm_kv = nn.GroupNorm(8, channels)

    def forward(self, query_features, context_features, reference_mask=None):
        """
        Args:
            query_features: 需要分类的特征 (B, C, H, W)
            context_features: 提供上下文的特征 (B, C, H, W)
            reference_mask: 已知肌肉区域mask (B, 1, H, W)，用于加权
        """
        B, C, H, W = query_features.shape
        N = H * W

        # 归一化
        q_norm = self.norm_q(query_features)
        kv_norm = self.norm_kv(context_features)

        # 生成Q (从query_features)
        q = self.q_proj(q_norm)  # (B, C, H, W)
        q = q.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, heads, N, head_dim)

        # 生成K, V (从context_features)
        kv = self.kv_proj(kv_norm)  # (B, 2C, H, W)
        kv = kv.reshape(B, 2, self.num_heads, self.head_dim, N)
        k = kv[:, 0].permute(0, 1, 3, 2)  # (B, heads, N, head_dim)
        v = kv[:, 1].permute(0, 1, 3, 2)  # (B, heads, N, head_dim)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # 如果有参考mask，加权已标注区域的重要性
        if reference_mask is not None:
            ref_flat = reference_mask.reshape(B, 1, 1, N)  # (B, 1, 1, N)
            # 增强对已标注区域的注意力
            attn = attn + ref_flat * 2.0  # 已标注区域的Key更重要

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out_proj(out)

        return query_features + out, attn


class SimilarityAttentionModule(nn.Module):
    """
    相似性注意力模块

    专门设计用于：
    1. 提取已标注肌肉区域的特征原型
    2. 计算未标注区域与原型的相似度
    3. 输出一个相似度图，指导分割
    """

    def __init__(self, channels, hidden_dim=128):
        super().__init__()

        # 特征压缩
        self.feature_compress = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 原型学习
        self.prototype_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 相似度输出
        self.similarity_head = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, features, muscle_mask):
        """
        Args:
            features: 特征图 (B, C, H, W)
            muscle_mask: 已知肌肉区域 (B, 1, H, W)

        Returns:
            similarity_map: 相似度图 (B, 1, H, W)
            prototype: 肌肉原型特征 (B, hidden_dim)
        """
        B, _, H, W = features.shape

        # 压缩特征
        feat = self.feature_compress(features)  # (B, hidden_dim, H, W)
        hidden_dim = feat.shape[1]

        # 提取已标注区域的特征（全局平均池化）
        muscle_mask_expanded = muscle_mask.expand_as(feat)
        masked_feat = feat * muscle_mask_expanded

        # 计算原型：已标注区域的平均特征
        num_pixels = muscle_mask.sum(dim=(2, 3), keepdim=True) + 1e-6  # (B, 1, 1, 1)
        prototype = masked_feat.sum(dim=(2, 3)) / num_pixels.squeeze(-1).squeeze(-1)  # (B, hidden_dim)

        # 精炼原型
        prototype = self.prototype_mlp(prototype)  # (B, hidden_dim)

        # 计算每个位置与原型的相似度
        prototype_expanded = prototype.view(B, hidden_dim, 1, 1).expand_as(feat)

        # 余弦相似度
        feat_norm = F.normalize(feat, dim=1)
        proto_norm = F.normalize(prototype_expanded, dim=1)
        similarity = (feat_norm * proto_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # 结合相似度生成最终的相似度图
        combined = torch.cat([feat, similarity], dim=1)
        similarity_map = self.similarity_head(combined)  # (B, 1, H, W)

        return similarity_map, prototype


# ============================================================================
# 3. 带注意力的UNet网络
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


class AttentionMuscleNet(nn.Module):
    """
    带注意力机制的肌肉分割网络

    主要改进：
    1. 位置编码：让网络感知空间位置
    2. 自注意力：在bottleneck层捕获全局上下文
    3. 跨区域注意力：以已标注区域为参考
    4. 相似性注意力：学习肌肉特征原型并匹配
    """

    def __init__(self, in_ch=4, features=[32, 64, 128, 256], num_heads=8):
        super().__init__()

        self.features = features

        # 位置编码
        self.pos_encoding = PositionalEncoding2D(features[0])

        # 编码器
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        current_ch = in_ch
        for f in features:
            self.encoder.append(DoubleConv(current_ch, f, dropout=0.1))
            current_ch = f

        # Bottleneck
        bottleneck_ch = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_ch, dropout=0.2)

        # Bottleneck自注意力
        self.bottleneck_attention = MultiHeadSelfAttention2D(bottleneck_ch, num_heads=num_heads)

        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        rev_features = list(reversed(features))
        up_ch = bottleneck_ch

        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(up_ch, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f, dropout=0.1))
            up_ch = f

        # 跨区域注意力（在解码器中间层）
        self.cross_attention = CrossRegionAttention(features[1], num_heads=4)  # 64通道层

        # 相似性注意力模块
        self.similarity_attention = SimilarityAttentionModule(features[0], hidden_dim=64)

        # 最终输出
        self.final = nn.Sequential(
            nn.Conv2d(features[0] + 1, features[0], 3, padding=1),  # +1 for similarity map
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], 1, 1),
        )

    def forward(self, x, muscle_mask=None):
        """
        Args:
            x: 输入 (B, 4, H, W) - CT, HU粗分割, 已知标签, 非排除区域
            muscle_mask: 已知肌肉标签 (B, 1, H, W)，用于注意力计算
        """
        # 如果没有提供muscle_mask，从输入的第3通道提取
        if muscle_mask is None:
            muscle_mask = x[:, 2:3, :, :]

        # 编码器
        skips = []
        out = x
        for i, enc in enumerate(self.encoder):
            out = enc(out)
            if i == 0:
                out = self.pos_encoding(out)  # 第一层加位置编码
            skips.append(out)
            out = self.pool(out)

        # Bottleneck + 自注意力
        out = self.bottleneck(out)
        out = self.bottleneck_attention(out)

        # 解码器
        skips = skips[::-1]
        cross_attn_map = None

        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoder)):
            out = upconv(out)
            skip = skips[i]

            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=True)

            out = torch.cat([skip, out], dim=1)
            out = dec(out)

            # 在64通道层应用跨区域注意力
            if i == len(self.upconvs) - 2:  # 倒数第二层
                # 下采样muscle_mask以匹配当前分辨率
                mask_scaled = F.interpolate(muscle_mask, size=out.shape[2:], mode='nearest')
                out, cross_attn_map = self.cross_attention(out, out, mask_scaled)

        # 相似性注意力
        similarity_map, prototype = self.similarity_attention(out, muscle_mask)

        # 合并相似度图
        out = torch.cat([out, torch.sigmoid(similarity_map)], dim=1)

        # 最终输出
        output = self.final(out)

        return output, similarity_map, cross_attn_map


# ============================================================================
# 4. 损失函数
# ============================================================================

class AttentionAwareLoss(nn.Module):
    """
    注意力感知损失函数

    在v3的基础上增加：
    1. 相似性一致性损失：相似度高的区域应该和已标注肌肉有相似的标签
    2. 原型对比损失：增强已标注/未标注区域的特征区分
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, similarity_map, label_mask, hu_coarse,
                exclusion_mask, body_mask):
        """
        Args:
            pred_logits: 模型主输出 (B, 1, H, W)
            similarity_map: 相似度图 (B, 1, H, W)
            label_mask: 已知肌肉标签 (B, 1, H, W)
            hu_coarse: HU粗分割 (B, 1, H, W)
            exclusion_mask: 排除区域 (B, 1, H, W)
            body_mask: 身体掩码 (B, 1, H, W)
        """
        pred_prob = torch.sigmoid(pred_logits)
        similarity_prob = torch.sigmoid(similarity_map)
        losses = {}

        # ============ 1. 已标注区域对齐损失 ============
        has_label = (label_mask > 0.5).float()
        bce_all = self.bce(pred_logits, label_mask)
        label_region_loss = (bce_all * has_label).sum() / (has_label.sum() + 1e-6)
        losses['label_alignment'] = label_region_loss * LABEL_ALIGNMENT_WEIGHT

        # ============ 2. 覆盖奖励 ============
        unlabeled_hu_valid = hu_coarse * (1 - has_label) * body_mask * (1 - exclusion_mask)
        coverage_loss = ((1 - pred_prob) * unlabeled_hu_valid).mean()
        losses['coverage_reward'] = coverage_loss * COVERAGE_REWARD_WEIGHT

        # ============ 3. 排除区域损失 ============
        if exclusion_mask.sum() > 0:
            exclusion_loss = (pred_prob * exclusion_mask).mean()
        else:
            exclusion_loss = torch.tensor(0.0, device=pred_logits.device)
        losses['exclusion'] = exclusion_loss * EXCLUSION_WEIGHT

        # ============ 4. HU范围外惩罚 ============
        hu_invalid = (1 - hu_coarse) * body_mask * (1 - exclusion_mask)
        hu_violation = (pred_prob * hu_invalid).mean()
        losses['hu_violation'] = hu_violation * 1.0

        # ============ 5. 边界平滑 ============
        tv_h = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]).mean()
        tv_w = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]).mean()
        losses['smoothness'] = (tv_h + tv_w) * SMOOTHNESS_WEIGHT

        # ============ 6. 相似性一致性损失（新增）============
        # 相似度高的区域应该预测为肌肉
        # 在未标注但HU合理的区域，相似度应该指导预测
        similarity_guidance = similarity_prob * unlabeled_hu_valid
        pred_in_similar = pred_prob * unlabeled_hu_valid

        # 相似度和预测应该一致
        similarity_consistency = F.mse_loss(pred_in_similar, similarity_guidance)
        losses['similarity_consistency'] = similarity_consistency * ATTENTION_SIMILARITY_WEIGHT

        # ============ 7. 相似度图监督（在已标注区域）============
        # 已标注区域的相似度应该高
        similarity_in_label = self.bce(similarity_map, label_mask)
        similarity_label_loss = (similarity_in_label * (has_label + unlabeled_hu_valid * 0.3)).mean()
        losses['similarity_supervision'] = similarity_label_loss * 0.5

        total_loss = sum(losses.values())
        losses['total'] = total_loss

        return total_loss, losses


# ============================================================================
# 5. 数据集（与v3相同）
# ============================================================================

class MuscleRefineDataset(Dataset):
    """肌肉精修数据集"""

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

        # HU粗分割
        hu_coarse = ((hu_resized >= MUSCLE_HU_MIN) & (hu_resized <= MUSCLE_HU_MAX)).astype(np.float32)

        # 排除区域
        exclusion = ((hu_resized < AIR_HU_MAX) | (hu_resized > BONE_HU_MIN)).astype(np.float32)

        # 身体掩码
        body_mask = self._get_body_mask(hu_resized)

        # 构建输入
        input_tensor = np.stack([
            ct_resized,
            hu_coarse,
            muscle_resized,
            1 - exclusion
        ], axis=0)

        input_t = torch.from_numpy(input_tensor).float()
        hu_coarse_t = torch.from_numpy(hu_coarse).unsqueeze(0).float()
        muscle_t = torch.from_numpy(muscle_resized).unsqueeze(0).float()
        body_t = torch.from_numpy(body_mask).unsqueeze(0).float()
        exclusion_t = torch.from_numpy(exclusion).unsqueeze(0).float()

        return input_t, hu_coarse_t, muscle_t, body_t, exclusion_t


# ============================================================================
# 6. 评估指标
# ============================================================================

def compute_metrics(pred, label, hu_coarse, body_mask, exclusion_mask):
    pred_binary = (pred > 0.5).float()
    label_binary = (label > 0.5).float()

    metrics = {}

    # 1. 标签区域召回率
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

    # 3. HU覆盖率
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
# 7. 可视化
# ============================================================================

def visualize_predictions(model, dataloader, device, epoch, output_dir, num_samples=4):
    model.eval()

    samples = []
    with torch.no_grad():
        for inputs, hu_coarse, labels, body_masks, exclusion_masks in dataloader:
            if len(samples) >= num_samples:
                break

            inputs = inputs.to(device)
            labels_dev = labels.to(device)
            pred_logits, similarity_map, _ = model(inputs, labels_dev)
            pred_prob = torch.sigmoid(pred_logits)
            similarity_prob = torch.sigmoid(similarity_map)

            for i in range(min(inputs.size(0), num_samples - len(samples))):
                samples.append({
                    'ct': inputs[i, 0].cpu().numpy(),
                    'hu_coarse': inputs[i, 1].cpu().numpy(),
                    'label': labels[i, 0].numpy(),
                    'pred': pred_prob[i, 0].cpu().numpy(),
                    'similarity': similarity_prob[i, 0].cpu().numpy(),
                    'exclusion': exclusion_masks[i, 0].numpy(),
                })
                if len(samples) >= num_samples:
                    break

    fig, axes = plt.subplots(num_samples, 7, figsize=(28, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample in enumerate(samples):
        ct = sample['ct']
        hu_coarse = sample['hu_coarse']
        label = sample['label']
        pred = sample['pred']
        similarity = sample['similarity']
        pred_binary = (pred > 0.5).astype(np.float32)

        # 1. CT原图
        axes[idx, 0].imshow(ct, cmap='gray')
        axes[idx, 0].set_title('CT Image')
        axes[idx, 0].axis('off')

        # 2. HU粗分割
        axes[idx, 1].imshow(ct, cmap='gray')
        axes[idx, 1].imshow(hu_coarse, cmap='Greens', alpha=0.4)
        axes[idx, 1].set_title('HU Coarse')
        axes[idx, 1].axis('off')

        # 3. 已知标签
        axes[idx, 2].imshow(ct, cmap='gray')
        axes[idx, 2].imshow(label, cmap='Reds', alpha=0.5)
        axes[idx, 2].set_title('Known Labels')
        axes[idx, 2].axis('off')

        # 4. 相似度图（新增）
        axes[idx, 3].imshow(ct, cmap='gray')
        axes[idx, 3].imshow(similarity, cmap='hot', alpha=0.6)
        axes[idx, 3].set_title('Similarity Map')
        axes[idx, 3].axis('off')

        # 5. 模型预测
        axes[idx, 4].imshow(ct, cmap='gray')
        axes[idx, 4].imshow(pred, cmap='Blues', alpha=0.5)
        axes[idx, 4].set_title('Prediction')
        axes[idx, 4].axis('off')

        # 6. 对比
        overlay = np.zeros((*ct.shape, 3))
        overlay[..., 0] = label * (1 - pred_binary)
        overlay[..., 1] = pred_binary * (1 - label)
        overlay[..., 2] = pred_binary * label
        axes[idx, 5].imshow(ct, cmap='gray')
        axes[idx, 5].imshow(overlay, alpha=0.6)
        axes[idx, 5].set_title('R:Miss G:New B:Match')
        axes[idx, 5].axis('off')

        # 7. 预测 vs HU
        overlay2 = np.zeros((*ct.shape, 3))
        overlay2[..., 0] = hu_coarse * (1 - pred_binary)
        overlay2[..., 1] = pred_binary * hu_coarse
        overlay2[..., 2] = pred_binary * (1 - hu_coarse)
        axes[idx, 6].imshow(ct, cmap='gray')
        axes[idx, 6].imshow(overlay2, alpha=0.6)
        axes[idx, 6].set_title('R:Missed G:Good B:Bad')
        axes[idx, 6].axis('off')

    plt.suptitle(f'Epoch {epoch} - Attention-based Muscle Transfer V4', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'visualization_epoch{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return wandb.Image(save_path, caption=f"Epoch {epoch}")


# ============================================================================
# 8. 主训练流程
# ============================================================================

def main():
    print("=" * 60)
    print("肌肉特征迁移学习 V4 - 注意力机制版本")
    print("=" * 60)

    DATA_ROOT = '/local/hzhang02/data'
    OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs_muscle_transfer_v4'
    TARGET_SHAPE = (256, 256)
    NUM_SUBJECTS = 50

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # wandb
    wandb.init(
        project="muscle-transfer-learning",
        name=f"transfer-v4-attention-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "version": "v4-attention",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "label_alignment_weight": LABEL_ALIGNMENT_WEIGHT,
            "coverage_reward_weight": COVERAGE_REWARD_WEIGHT,
            "exclusion_weight": EXCLUSION_WEIGHT,
            "smoothness_weight": SMOOTHNESS_WEIGHT,
            "attention_similarity_weight": ATTENTION_SIMILARITY_WEIGHT,
            "muscle_hu_range": [MUSCLE_HU_MIN, MUSCLE_HU_MAX],
            "abdominal_only": ABDOMINAL_ONLY,
        }
    )
    print("   wandb 初始化完成")

    # 模型
    print("\n1. 初始化模型...")
    model = AttentionMuscleNet(in_ch=4, features=[32, 64, 128, 256], num_heads=8)
    model.to(device)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = AttentionAwareLoss()
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
        loss_components = {
            'label_alignment': [], 'coverage_reward': [], 'exclusion': [],
            'hu_violation': [], 'smoothness': [], 'similarity_consistency': [],
            'similarity_supervision': []
        }

        for inputs, hu_coarse, labels, body_masks, exclusion_masks in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            hu_coarse = hu_coarse.to(device)
            labels = labels.to(device)
            body_masks = body_masks.to(device)
            exclusion_masks = exclusion_masks.to(device)

            optimizer.zero_grad()
            pred_logits, similarity_map, _ = model(inputs, labels)
            loss, loss_dict = criterion(pred_logits, similarity_map, labels,
                                        hu_coarse, exclusion_masks, body_masks)

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

                pred_logits, similarity_map, _ = model(inputs, labels)
                loss, _ = criterion(pred_logits, similarity_map, labels,
                                    hu_coarse, exclusion_masks, body_masks)
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
        print(f"  标签召回率: {avg_metrics['label_recall']:.4f}")
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
            "train/similarity_consistency": np.mean(loss_components['similarity_consistency']),
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
    axes[0, 1].set_title('Label Recall')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['epoch'], history['label_dice'])
    axes[0, 2].set_title('Label Dice')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history['epoch'], history['hu_coverage'])
    axes[1, 0].set_title('HU Coverage')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['epoch'], history['exclusion_overflow'])
    axes[1, 1].set_title('Exclusion Overflow')
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
