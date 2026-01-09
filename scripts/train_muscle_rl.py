#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强化学习二次训练 - 全身肌肉群分割

本脚本使用已训练好的10种肌肉模型作为教师，通过强化学习方法进行二次训练，
以识别整个样本的完整肌肉群（包括环腹部肌肉等未被原模型覆盖的区域）。

核心思想：
1. 教师模型提供10种已知肌肉的精确分割
2. 从教师模型预测中提取肌肉的HU值分布特征（"颜色"）
3. 使用策略梯度强化学习，让新模型学习：
   - 已知肌肉区域：与教师模型一致（正奖励）
   - 未知区域：基于学到的HU特征进行探索（探索奖励）
4. 最终获得能够分割完整肌肉群的模型

强化学习环境设计：
- 状态 (State): CT切片 + 位置编码 + 教师模型预测
- 动作 (Action): 像素级二值分割决策 (肌肉/非肌肉)
- 奖励 (Reward):
  * 教师区域一致性奖励
  * HU范围合规奖励
  * 空间连续性奖励
  * 形态学合理性奖励

作者: hzhang02
日期: 2025-01-09
"""

import os
import json
import glob
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
from scipy import ndimage, stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.distributions import Bernoulli
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

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

# 肌肉HU值范围（从已训练模型中学习到的特征）
MUSCLE_HU_MIN = -29
MUSCLE_HU_MAX = 150
BODY_HU_MIN = -500

# 强化学习超参数
GAMMA = 0.99  # 折扣因子
ENTROPY_COEF = 0.01  # 熵正则化系数
VALUE_COEF = 0.5  # 价值函数损失系数
CLIP_EPSILON = 0.2  # PPO裁剪参数
GAE_LAMBDA = 0.95  # GAE参数

# 奖励权重
TEACHER_CONSISTENCY_WEIGHT = 1.0  # 与教师模型一致的奖励
HU_RANGE_WEIGHT = 0.5  # HU值在肌肉范围内的奖励
SPATIAL_CONTINUITY_WEIGHT = 0.3  # 空间连续性奖励
MORPHOLOGY_WEIGHT = 0.2  # 形态学合理性奖励


# ============================================================================
# 2. 教师模型定义 (与原训练脚本一致)
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
# 3. 强化学习策略网络 (Actor-Critic)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    策略网络 (Actor-Critic)

    输入: CT图像 + 教师模型预测 + 位置编码
    输出:
        - policy: 每个像素是肌肉的概率
        - value: 状态价值估计
    """

    def __init__(self, in_ch=3, features=[32, 64, 128, 256]):
        """
        Args:
            in_ch: 输入通道数
                   1 (CT) + 1 (教师预测) + 1 (HU mask) = 3
        """
        super().__init__()

        # 共享特征提取器 (编码器)
        self.encoder = nn.ModuleList()
        current_ch = in_ch
        for f in features:
            self.encoder.append(DoubleConv(current_ch, f))
            current_ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 策略头 (Actor) - 解码器
        self.policy_decoder = nn.ModuleList()
        rev = list(reversed(features))
        up_in = features[-1] * 2
        for f in rev:
            self.policy_decoder.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            self.policy_decoder.append(DoubleConv(up_in, f))
            up_in = f
        self.policy_head = nn.Conv2d(features[0], 1, kernel_size=1)

        # 价值头 (Critic) - 全局池化 + MLP
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_head = nn.Sequential(
            nn.Linear(features[-1] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 编码器
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        bottleneck_features = self.bottleneck(x)

        # 策略解码器
        policy_x = bottleneck_features
        for idx in range(0, len(self.policy_decoder), 2):
            trans = self.policy_decoder[idx]
            conv = self.policy_decoder[idx + 1]
            policy_x = trans(policy_x)
            skip = skips[-(idx // 2) - 1]
            if policy_x.shape != skip.shape:
                _, _, h, w = policy_x.shape
                skip = skip[:, :, :h, :w]
            policy_x = torch.cat([skip, policy_x], dim=1)
            policy_x = conv(policy_x)

        # 输出策略 (像素级概率)
        policy_logits = self.policy_head(policy_x)

        # 价值估计
        value_features = self.value_pool(bottleneck_features)
        value_features = value_features.view(value_features.size(0), -1)
        value = self.value_head(value_features)

        return policy_logits, value


# ============================================================================
# 4. 肌肉特征提取器 (从教师模型学习)
# ============================================================================

class MuscleFeatureExtractor:
    """
    从已训练的教师模型中提取肌肉的特征分布

    这些特征包括：
    - HU值分布（均值、标准差、范围）
    - 纹理特征
    - 空间分布模式
    """

    def __init__(self, teacher_model, device):
        self.teacher_model = teacher_model
        self.device = device
        self.muscle_hu_stats = {}  # 每种肌肉的HU统计
        self.global_muscle_hu = {
            'mean': (MUSCLE_HU_MIN + MUSCLE_HU_MAX) / 2,
            'std': (MUSCLE_HU_MAX - MUSCLE_HU_MIN) / 4,
            'min': MUSCLE_HU_MIN,
            'max': MUSCLE_HU_MAX
        }

    def extract_features_from_slice(self, ct_slice, original_hu_slice):
        """
        从单个切片提取肌肉特征

        Args:
            ct_slice: 归一化的CT切片 (H, W)
            original_hu_slice: 原始HU值切片 (H, W)

        Returns:
            dict: 肌肉特征
        """
        H, W = ct_slice.shape
        target_shape = (256, 256)

        # 调整大小
        ct_resized = resize(ct_slice, target_shape, order=1, preserve_range=True)

        # 通过教师模型预测
        img_t = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            pred = torch.sigmoid(self.teacher_model(img_t))
            pred_combined = (pred[0].sum(dim=0) > 0.5).cpu().numpy()

        # 调整回原始大小
        pred_original = resize(pred_combined, (H, W), order=0, preserve_range=True)
        pred_mask = pred_original > 0.5

        # 提取肌肉区域的HU特征
        if pred_mask.sum() > 0:
            muscle_hu_values = original_hu_slice[pred_mask]
            features = {
                'hu_mean': np.mean(muscle_hu_values),
                'hu_std': np.std(muscle_hu_values),
                'hu_min': np.percentile(muscle_hu_values, 5),
                'hu_max': np.percentile(muscle_hu_values, 95),
                'area_ratio': pred_mask.sum() / pred_mask.size,
                'teacher_mask': pred_mask
            }
        else:
            features = {
                'hu_mean': self.global_muscle_hu['mean'],
                'hu_std': self.global_muscle_hu['std'],
                'hu_min': self.global_muscle_hu['min'],
                'hu_max': self.global_muscle_hu['max'],
                'area_ratio': 0,
                'teacher_mask': pred_mask
            }

        return features

    def update_global_stats(self, all_features):
        """更新全局肌肉HU统计"""
        all_means = [f['hu_mean'] for f in all_features if f['area_ratio'] > 0]
        if all_means:
            self.global_muscle_hu['mean'] = np.mean(all_means)
            self.global_muscle_hu['std'] = np.mean([f['hu_std'] for f in all_features if f['area_ratio'] > 0])


# ============================================================================
# 5. 强化学习环境
# ============================================================================

class MuscleSegmentationEnv:
    """
    肌肉分割强化学习环境

    设计思路：
    - 智能体需要决定每个像素是否属于肌肉
    - 奖励基于：教师一致性、HU合规性、空间连续性、形态学合理性
    """

    def __init__(self, teacher_model, feature_extractor, device):
        self.teacher_model = teacher_model
        self.feature_extractor = feature_extractor
        self.device = device
        self.target_shape = (256, 256)

    def compute_reward(self, action_mask, ct_slice, hu_slice, teacher_mask):
        """
        计算奖励

        Args:
            action_mask: 策略网络的分割决策 (H, W) numpy array
            ct_slice: 归一化CT切片 (H, W)
            hu_slice: 原始HU值切片 (H, W)
            teacher_mask: 教师模型预测 (H, W)

        Returns:
            float: 总奖励
            dict: 各项奖励分解
        """
        rewards = {}

        # 1. 教师一致性奖励
        # 在教师预测为肌肉的区域，动作应该一致
        teacher_region = teacher_mask > 0.5
        if teacher_region.sum() > 0:
            agreement = (action_mask[teacher_region] == teacher_mask[teacher_region]).mean()
            rewards['teacher_consistency'] = agreement * TEACHER_CONSISTENCY_WEIGHT
        else:
            rewards['teacher_consistency'] = 0

        # 2. HU范围奖励
        # 被预测为肌肉的区域，HU值应该在肌肉范围内
        predicted_muscle = action_mask > 0.5
        if predicted_muscle.sum() > 0:
            muscle_hu = hu_slice[predicted_muscle]
            in_range = ((muscle_hu >= MUSCLE_HU_MIN) & (muscle_hu <= MUSCLE_HU_MAX)).mean()
            rewards['hu_range'] = in_range * HU_RANGE_WEIGHT
        else:
            rewards['hu_range'] = 0

        # 3. 空间连续性奖励
        # 肌肉区域应该是连续的，不应该有太多小碎片
        if predicted_muscle.sum() > 0:
            labeled = label(predicted_muscle)
            if labeled.max() > 0:
                regions = regionprops(labeled)
                # 惩罚小区域
                valid_regions = [r for r in regions if r.area > 50]
                continuity = len(valid_regions) / max(len(regions), 1)
                rewards['spatial_continuity'] = continuity * SPATIAL_CONTINUITY_WEIGHT
            else:
                rewards['spatial_continuity'] = 0
        else:
            rewards['spatial_continuity'] = 0

        # 4. 形态学合理性奖励
        # 肌肉形状应该是合理的（不是细长线条）
        if predicted_muscle.sum() > 100:
            # 计算紧凑度
            labeled = label(predicted_muscle)
            if labeled.max() > 0:
                regions = regionprops(labeled)
                compactness_scores = []
                for r in regions:
                    if r.area > 50:
                        # 紧凑度 = 4 * pi * area / perimeter^2
                        if r.perimeter > 0:
                            compactness = 4 * np.pi * r.area / (r.perimeter ** 2)
                            compactness_scores.append(min(compactness, 1.0))
                if compactness_scores:
                    rewards['morphology'] = np.mean(compactness_scores) * MORPHOLOGY_WEIGHT
                else:
                    rewards['morphology'] = 0
            else:
                rewards['morphology'] = 0
        else:
            rewards['morphology'] = 0

        # 5. 探索奖励 (鼓励在教师未覆盖的肌肉区域进行探索)
        non_teacher_region = ~teacher_region
        hu_in_muscle_range = (hu_slice >= MUSCLE_HU_MIN) & (hu_slice <= MUSCLE_HU_MAX)
        potential_muscle = non_teacher_region & hu_in_muscle_range

        if potential_muscle.sum() > 0:
            # 在潜在肌肉区域的预测
            exploration_coverage = (action_mask[potential_muscle] > 0.5).mean()
            rewards['exploration'] = exploration_coverage * 0.3
        else:
            rewards['exploration'] = 0

        total_reward = sum(rewards.values())
        return total_reward, rewards

    def get_body_mask(self, hu_slice):
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


# ============================================================================
# 6. 强化学习数据集
# ============================================================================

class RLMuscleDataset(Dataset):
    """
    强化学习训练数据集

    返回: CT切片, HU切片, 教师预测, 身体掩码
    """

    def __init__(self, subjects, teacher_model, feature_extractor, device, target_shape=(256, 256)):
        self.items = []
        self.target_shape = target_shape
        self.teacher_model = teacher_model
        self.feature_extractor = feature_extractor
        self.device = device

        print("构建强化学习数据集...")
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

        # 加载CT
        img = nib.load(ct_path).get_fdata().astype(np.float32)
        hu_slice = img[:, :, z]  # 原始HU值
        original_shape = hu_slice.shape

        # 归一化
        lo, hi = np.percentile(hu_slice, 1), np.percentile(hu_slice, 99)
        ct_normalized = np.clip(hu_slice, lo, hi)
        if hi - lo > 0:
            ct_normalized = (ct_normalized - lo) / (hi - lo)
        else:
            ct_normalized = np.zeros_like(ct_normalized)

        # 获取教师预测
        features = self.feature_extractor.extract_features_from_slice(ct_normalized, hu_slice)
        teacher_mask = features['teacher_mask'].astype(np.float32)

        # HU范围掩码 (在肌肉HU范围内)
        hu_muscle_mask = ((hu_slice >= MUSCLE_HU_MIN) & (hu_slice <= MUSCLE_HU_MAX)).astype(np.float32)

        # 调整大小
        H, W = self.target_shape
        ct_resized = resize(ct_normalized, (H, W), order=1, preserve_range=True)
        hu_resized = resize(hu_slice, (H, W), order=1, preserve_range=True)
        teacher_resized = resize(teacher_mask, (H, W), order=0, preserve_range=True)
        hu_muscle_resized = resize(hu_muscle_mask, (H, W), order=0, preserve_range=True)

        # 构建输入 (3通道: CT + 教师预测 + HU掩码)
        input_tensor = np.stack([ct_resized, teacher_resized, hu_muscle_resized], axis=0)

        # 转换为tensor
        input_t = torch.from_numpy(input_tensor).float()
        hu_t = torch.from_numpy(hu_resized).float()
        teacher_t = torch.from_numpy(teacher_resized).float()

        return input_t, hu_t, teacher_t


# ============================================================================
# 7. PPO训练器
# ============================================================================

class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) 训练器
    """

    def __init__(self, policy_net, env, lr=3e-4):
        self.policy_net = policy_net
        self.env = env
        self.optimizer = Adam(policy_net.parameters(), lr=lr)

    def compute_returns_and_advantages(self, rewards, values, dones):
        """
        计算回报和优势函数 (使用GAE)
        """
        returns = []
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        PPO更新步骤
        """
        # 前向传播
        policy_logits, values = self.policy_net(states)

        # 策略损失
        probs = torch.sigmoid(policy_logits)
        dist = Bernoulli(probs)
        new_log_probs = dist.log_prob(actions).sum(dim=[1, 2, 3])

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(values.squeeze(), returns)

        # 熵损失 (鼓励探索)
        entropy = dist.entropy().mean()

        # 总损失
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

        return {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


# ============================================================================
# 8. 评估指标
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
# 9. 主训练流程
# ============================================================================

def main():
    print("=" * 60)
    print("强化学习二次训练 - 全身肌肉群分割")
    print("=" * 60)

    # 配置
    DATA_ROOT = '/local/hzhang02/data'
    OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs_rl_muscle'
    TEACHER_MODEL_PATH = '/local/hzhang02/data/dataset/outputs/best_model.pth'
    LABEL_MAP_PATH = '/local/hzhang02/data/dataset/outputs/label_map.json'

    TARGET_SHAPE = (256, 256)
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    EPOCHS = 10
    NUM_SUBJECTS = 50
    EPISODES_PER_EPOCH = 100

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== 加载教师模型 ==========
    print("\n1. 加载教师模型...")

    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"   教师模型分割 {num_classes} 种肌肉")

    teacher_model = UNet2D(in_ch=1, out_ch=num_classes, features=[32, 64, 128, 256])
    checkpoint = torch.load(TEACHER_MODEL_PATH, map_location=device)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.to(device)
    teacher_model.eval()
    print(f"   教师模型加载成功: {TEACHER_MODEL_PATH}")
    print(f"   教师模型验证Dice: {checkpoint.get('val_dice', 'N/A'):.4f}")

    # ========== 初始化组件 ==========
    print("\n2. 初始化强化学习组件...")

    # 特征提取器
    feature_extractor = MuscleFeatureExtractor(teacher_model, device)
    print("   特征提取器已初始化")

    # 环境
    env = MuscleSegmentationEnv(teacher_model, feature_extractor, device)
    print("   强化学习环境已初始化")

    # 策略网络 (3通道输入: CT + 教师预测 + HU掩码)
    policy_net = PolicyNetwork(in_ch=3, features=[32, 64, 128, 256]).to(device)
    print(f"   策略网络参数量: {sum(p.numel() for p in policy_net.parameters()):,}")

    # PPO训练器
    trainer = PPOTrainer(policy_net, env, lr=LEARNING_RATE)
    print("   PPO训练器已初始化")

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
    train_ds = RLMuscleDataset(train_subs, teacher_model, feature_extractor, device, TARGET_SHAPE)
    val_ds = RLMuscleDataset(val_subs, teacher_model, feature_extractor, device, TARGET_SHAPE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"   训练切片: {len(train_ds)} / 验证切片: {len(val_ds)}")

    # ========== 训练循环 ==========
    print("\n4. 开始强化学习训练...")
    print("=" * 60)

    history = {
        'epoch': [],
        'avg_reward': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'val_coverage': [],
        'val_hu_compliance': []
    }

    best_val_coverage = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)

        # 训练阶段
        policy_net.train()
        epoch_rewards = []
        epoch_losses = {'policy': [], 'value': [], 'entropy': []}

        for batch_idx, (inputs, hu_slices, teacher_masks) in enumerate(tqdm(train_loader, desc="Training")):
            inputs = inputs.to(device)
            hu_slices = hu_slices.numpy()
            teacher_masks = teacher_masks.numpy()

            # 前向传播获取策略
            policy_logits, values = policy_net(inputs)
            probs = torch.sigmoid(policy_logits)

            # 采样动作
            dist = Bernoulli(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=[1, 2, 3])

            # 计算每个样本的奖励
            batch_rewards = []
            for i in range(inputs.size(0)):
                action_mask = actions[i, 0].cpu().numpy()
                reward, _ = env.compute_reward(
                    action_mask,
                    inputs[i, 0].cpu().numpy(),
                    hu_slices[i],
                    teacher_masks[i]
                )
                batch_rewards.append(reward)

            rewards_tensor = torch.tensor(batch_rewards, device=device)
            epoch_rewards.extend(batch_rewards)

            # 计算优势
            values_np = values.squeeze().detach().cpu().numpy()
            returns, advantages = trainer.compute_returns_and_advantages(
                batch_rewards,
                values_np.tolist() if len(values_np.shape) > 0 else [values_np.item()],
                [0] * len(batch_rewards)
            )
            returns = returns.to(device)
            advantages = advantages.to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO更新
            loss_info = trainer.update(
                inputs, actions, log_probs.detach(),
                returns, advantages
            )

            epoch_losses['policy'].append(loss_info['policy_loss'])
            epoch_losses['value'].append(loss_info['value_loss'])
            epoch_losses['entropy'].append(loss_info['entropy'])

        # 验证阶段
        policy_net.eval()
        val_coverages = []
        val_hu_compliances = []

        with torch.no_grad():
            for inputs, hu_slices, teacher_masks in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                hu_slices = hu_slices.numpy()

                policy_logits, _ = policy_net(inputs)
                preds = (torch.sigmoid(policy_logits) > 0.5).float()

                for i in range(inputs.size(0)):
                    pred_mask = preds[i, 0].cpu().numpy()
                    hu_slice = hu_slices[i]

                    # 计算覆盖率 (预测的肌肉区域占身体的比例)
                    body_mask = env.get_body_mask(hu_slice)
                    if body_mask.sum() > 0:
                        coverage = pred_mask[body_mask > 0.5].mean()
                        val_coverages.append(coverage)

                    # 计算HU合规率
                    if pred_mask.sum() > 0:
                        muscle_hu = hu_slice[pred_mask > 0.5]
                        compliance = ((muscle_hu >= MUSCLE_HU_MIN) &
                                      (muscle_hu <= MUSCLE_HU_MAX)).mean()
                        val_hu_compliances.append(compliance)

        # 记录结果
        avg_reward = np.mean(epoch_rewards)
        avg_policy_loss = np.mean(epoch_losses['policy'])
        avg_value_loss = np.mean(epoch_losses['value'])
        avg_entropy = np.mean(epoch_losses['entropy'])
        avg_coverage = np.mean(val_coverages) if val_coverages else 0
        avg_hu_compliance = np.mean(val_hu_compliances) if val_hu_compliances else 0

        history['epoch'].append(epoch)
        history['avg_reward'].append(avg_reward)
        history['policy_loss'].append(avg_policy_loss)
        history['value_loss'].append(avg_value_loss)
        history['entropy'].append(avg_entropy)
        history['val_coverage'].append(avg_coverage)
        history['val_hu_compliance'].append(avg_hu_compliance)

        epoch_time = time.time() - epoch_start

        print(f"\n结果:")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  策略损失: {avg_policy_loss:.4f}")
        print(f"  价值损失: {avg_value_loss:.4f}")
        print(f"  熵: {avg_entropy:.4f}")
        print(f"  验证覆盖率: {avg_coverage:.4f}")
        print(f"  验证HU合规率: {avg_hu_compliance:.4f}")
        print(f"  耗时: {epoch_time:.1f}秒")

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'avg_reward': avg_reward,
            'val_coverage': avg_coverage,
        }, os.path.join(OUTPUT_DIR, f'checkpoint_rl_epoch{epoch}.pth'))

        # 保存最佳模型
        if avg_coverage > best_val_coverage:
            best_val_coverage = avg_coverage
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_net.state_dict(),
                'val_coverage': avg_coverage,
                'val_hu_compliance': avg_hu_compliance,
            }, os.path.join(OUTPUT_DIR, 'best_rl_muscle_model.pth'))
            print(f"  新最佳模型！覆盖率: {best_val_coverage:.4f}")

    # ========== 保存结果 ==========
    print("\n5. 保存训练结果...")

    # 保存历史
    with open(os.path.join(OUTPUT_DIR, 'training_history_rl.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 可视化训练曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(history['epoch'], history['avg_reward'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Average Reward per Epoch')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['epoch'], history['policy_loss'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Policy Loss')
    axes[0, 1].set_title('Policy Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['epoch'], history['value_loss'])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Value Loss')
    axes[0, 2].set_title('Value Loss')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history['epoch'], history['entropy'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Policy Entropy')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['epoch'], history['val_coverage'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Coverage')
    axes[1, 1].set_title('Validation Muscle Coverage')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history['epoch'], history['val_hu_compliance'])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('HU Compliance')
    axes[1, 2].set_title('Validation HU Compliance')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_rl.png'), dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("强化学习训练完成！")
    print(f"  最佳验证覆盖率: {best_val_coverage:.4f}")
    print(f"  结果保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
