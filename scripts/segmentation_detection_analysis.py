#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D医学图像分割检测与评估系统

本脚本用于检测和评估3D CT扫描的分割模型性能，包括：
- 数据加载与探索
- 模型推理与预测
- 评估指标计算（Dice系数、IoU、成功率）
- 2D/3D可视化
- 综合统计报告
"""

# ============================================================================
# 1. 环境设置与依赖导入
# ============================================================================

import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from skimage.measure import marching_cubes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 2. U-Net模型定义
# ============================================================================

class UNet2D(nn.Module):
    """2D U-Net模型用于医学图像分割"""

    def __init__(self, in_ch=1, out_ch=117, features=[32, 64, 128, 256]):
        super(UNet2D, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        in_channels = in_ch
        for feature in features:
            self.encoders.append(self._block(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = self._block(features[-1], features[-1]*2)

        # 解码器
        for feature in reversed(features):
            self.decoders.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(self._block(feature*2, feature))

        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码路径
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码路径
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[idx+1](x)

        return self.final_conv(x)


print("U-Net模型定义完成")


# ============================================================================
# 3. 数据集类定义
# ============================================================================

class CTSegmentationDataset(Dataset):
    """CT图像分割数据集"""

    def __init__(self, data_dir, subject_ids, label_map=None, target_shape=(256, 256)):
        self.data_dir = data_dir
        self.subject_ids = subject_ids
        self.target_shape = target_shape
        self.samples = []

        # 如果没有提供label_map，自动生成
        if label_map is None:
            seg_files = sorted(os.listdir(os.path.join(data_dir, subject_ids[0], 'segmentations')))
            self.label_map = {f: i for i, f in enumerate(seg_files)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

        # 收集所有切片
        for subj_id in subject_ids:
            ct_path = os.path.join(data_dir, subj_id, 'ct.nii.gz')
            if os.path.exists(ct_path):
                ct_img = nib.load(ct_path)
                num_slices = ct_img.shape[2]
                for slice_idx in range(num_slices):
                    self.samples.append((subj_id, slice_idx))

    def __len__(self):
        return len(self.samples)

    def _normalize_ct(self, ct_slice):
        """使用百分位数归一化CT切片"""
        p1, p99 = np.percentile(ct_slice, (1, 99))
        ct_slice = np.clip(ct_slice, p1, p99)
        ct_slice = (ct_slice - p1) / (p99 - p1 + 1e-8)
        return ct_slice

    def __getitem__(self, idx):
        subj_id, slice_idx = self.samples[idx]

        # 加载CT切片
        ct_path = os.path.join(self.data_dir, subj_id, 'ct.nii.gz')
        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata()
        ct_slice = ct_data[:, :, slice_idx]

        # 归一化和调整大小
        ct_slice = self._normalize_ct(ct_slice)
        ct_slice = resize(ct_slice, self.target_shape, anti_aliasing=True, preserve_range=True)

        # 加载分割掩码
        seg_dir = os.path.join(self.data_dir, subj_id, 'segmentations')
        mask = np.zeros((self.num_classes,) + self.target_shape, dtype=np.float32)

        for seg_file, channel_idx in self.label_map.items():
            seg_path = os.path.join(seg_dir, seg_file)
            if os.path.exists(seg_path):
                seg_img = nib.load(seg_path)
                seg_data = seg_img.get_fdata()
                seg_slice = seg_data[:, :, slice_idx]
                seg_slice = resize(seg_slice, self.target_shape, order=0, anti_aliasing=False, preserve_range=True)
                mask[channel_idx] = (seg_slice > 0).astype(np.float32)

        # 转换为tensor
        ct_tensor = torch.from_numpy(ct_slice).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).float()

        return ct_tensor, mask_tensor, subj_id, slice_idx


print("数据集类定义完成")


# ============================================================================
# 4. 评估指标函数
# ============================================================================

def dice_coefficient(pred, target, epsilon=1e-7):
    """
    计算Dice系数

    Args:
        pred: 预测掩码 (B, C, H, W) 或 (C, H, W)
        target: 真实掩码 (B, C, H, W) 或 (C, H, W)
        epsilon: 平滑项

    Returns:
        Dice系数 (每个类别一个值)
    """
    pred = (pred > 0.5).float()

    # 如果是4D张量，在batch维度上求平均
    if pred.dim() == 4:
        dims = (0, 2, 3)
    else:
        dims = (1, 2)

    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice


def iou_score(pred, target, epsilon=1e-7):
    """
    计算IoU (Intersection over Union)

    Args:
        pred: 预测掩码 (B, C, H, W) 或 (C, H, W)
        target: 真实掩码 (B, C, H, W) 或 (C, H, W)
        epsilon: 平滑项

    Returns:
        IoU分数 (每个类别一个值)
    """
    pred = (pred > 0.5).float()

    if pred.dim() == 4:
        dims = (0, 2, 3)
    else:
        dims = (1, 2)

    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection

    iou = (intersection + epsilon) / (union + epsilon)
    return iou


def calculate_success_rate(dice_scores, threshold=0.7):
    """
    计算检测成功率（Dice系数高于阈值的比例）

    Args:
        dice_scores: Dice系数数组
        threshold: 成功阈值

    Returns:
        成功率 (0-1)
    """
    return (dice_scores > threshold).float().mean().item()


def hausdorff_distance_2d(pred, target):
    """
    计算简化的2D Hausdorff距离
    （仅计算边界点之间的最大距离）
    """
    from scipy.ndimage import binary_erosion

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # 提取边界
    pred_boundary = pred_np ^ binary_erosion(pred_np)
    target_boundary = target_np ^ binary_erosion(target_np)

    # 如果没有边界点，返回0
    if not pred_boundary.any() or not target_boundary.any():
        return 0.0

    # 获取边界点坐标
    pred_points = np.argwhere(pred_boundary)
    target_points = np.argwhere(target_boundary)

    # 计算Hausdorff距离
    from scipy.spatial.distance import cdist
    distances_pred_to_target = cdist(pred_points, target_points).min(axis=1).max()
    distances_target_to_pred = cdist(target_points, pred_points).min(axis=1).max()

    return max(distances_pred_to_target, distances_target_to_pred)


print("评估指标函数定义完成")


# ============================================================================
# 5. 数据探索与统计
# ============================================================================

# 设置数据目录
DATA_DIR = '/local/hzhang02/data'
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取所有受试者ID，只使用 s0000 到 s0100
all_subjects_raw = [d for d in os.listdir(DATA_DIR) if d.startswith('s') and os.path.isdir(os.path.join(DATA_DIR, d))]
all_subjects = [s for s in all_subjects_raw if s in [f's{i:04d}' for i in range(101)]]
all_subjects = sorted(all_subjects)

print(f"找到 {len(all_subjects)} 个受试者: {all_subjects}")

# 统计信息
print("\n=== 数据集统计信息 ===")
for subj_id in all_subjects:
    ct_path = os.path.join(DATA_DIR, subj_id, 'ct.nii.gz')
    seg_dir = os.path.join(DATA_DIR, subj_id, 'segmentations')

    if os.path.exists(ct_path):
        ct_img = nib.load(ct_path)
        shape = ct_img.shape
        num_segs = len(os.listdir(seg_dir)) if os.path.exists(seg_dir) else 0
        print(f"{subj_id}: CT形状={shape}, 分割数量={num_segs}")

# 加载标签映射
label_map_path = os.path.join(OUTPUT_DIR, 'label_map.json')

if os.path.exists(label_map_path):
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    print(f"\n从 {label_map_path} 加载标签映射")
else:
    # 创建新的标签映射
    seg_dir = os.path.join(DATA_DIR, all_subjects[0], 'segmentations')
    seg_files = sorted(os.listdir(seg_dir))
    label_map = {f: i for i, f in enumerate(seg_files)}

    # 保存标签映射
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"\n创建并保存标签映射到 {label_map_path}")

print(f"\n解剖结构数量: {len(label_map)}")
print(f"\n前10个解剖结构:")
for i, (name, idx) in enumerate(list(label_map.items())[:10]):
    print(f"  {idx}: {name.replace('.nii.gz', '')}")


# ============================================================================
# 6. 模型加载与推理
# ============================================================================

# 查找最新的检查点
checkpoint_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('checkpoint_') and f.endswith('.pth')]

if checkpoint_files:
    # 按epoch排序，选择最新的
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0].replace('epoch', '')))
    latest_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(OUTPUT_DIR, latest_checkpoint)

    print(f"找到检查点: {latest_checkpoint}")

    # 创建模型
    model = UNet2D(in_ch=1, out_ch=len(label_map), features=[32, 64, 128, 256])

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"成功加载模型，epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_dice' in checkpoint:
        print(f"验证集Dice系数: {checkpoint['val_dice']:.4f}")

    model_loaded = True
else:
    print("未找到训练好的模型检查点")
    print("请先运行 train_unet.py 训练模型")
    model_loaded = False


# ============================================================================
# 7. 分割检测与评估
# ============================================================================

if model_loaded:
    # 选择一个测试受试者
    test_subject = all_subjects[0]

    print(f"\n=== 对受试者 {test_subject} 进行分割检测 ===")

    # 创建数据集
    test_dataset = CTSegmentationDataset(
        DATA_DIR,
        [test_subject],
        label_map=label_map,
        target_shape=(256, 256)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    print(f"测试切片数量: {len(test_dataset)}")

    # 存储所有预测和指标
    all_dice_scores = []
    all_iou_scores = []
    all_predictions = []
    all_targets = []
    all_images = []

    # 推理
    with torch.no_grad():
        for batch_idx, (images, masks, subj_ids, slice_indices) in enumerate(tqdm(test_loader, desc="推理中")):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            predictions = torch.sigmoid(outputs)

            # 计算指标
            dice = dice_coefficient(predictions, masks)
            iou = iou_score(predictions, masks)

            all_dice_scores.append(dice.cpu())
            all_iou_scores.append(iou.cpu())

            # 保存部分样本用于可视化
            if batch_idx < 5:
                all_predictions.append(predictions.cpu())
                all_targets.append(masks.cpu())
                all_images.append(images.cpu())

    # 合并所有指标
    all_dice_scores = torch.cat(all_dice_scores, dim=0)
    all_iou_scores = torch.cat(all_iou_scores, dim=0)

    print("\n推理完成！")


# ============================================================================
# 8. 评估指标统计与分析
# ============================================================================

if model_loaded:
    # 计算每个类别的平均指标
    mean_dice_per_class = all_dice_scores.mean(dim=0)
    mean_iou_per_class = all_iou_scores.mean(dim=0)
    std_dice_per_class = all_dice_scores.std(dim=0)
    std_iou_per_class = all_iou_scores.std(dim=0)

    # 总体指标
    overall_dice = mean_dice_per_class.mean().item()
    overall_iou = mean_iou_per_class.mean().item()

    # 成功率（使用不同阈值）
    success_rate_50 = calculate_success_rate(mean_dice_per_class, threshold=0.5)
    success_rate_70 = calculate_success_rate(mean_dice_per_class, threshold=0.7)
    success_rate_80 = calculate_success_rate(mean_dice_per_class, threshold=0.8)

    print("\n=== 总体评估指标 ===")
    print(f"平均Dice系数: {overall_dice:.4f}")
    print(f"平均IoU分数: {overall_iou:.4f}")
    print(f"\n成功率 (Dice > 0.5): {success_rate_50:.2%}")
    print(f"成功率 (Dice > 0.7): {success_rate_70:.2%}")
    print(f"成功率 (Dice > 0.8): {success_rate_80:.2%}")

    # 创建详细的结果DataFrame
    results_df = pd.DataFrame({
        '解剖结构': [k.replace('.nii.gz', '') for k in label_map.keys()],
        'Dice系数': mean_dice_per_class.numpy(),
        'Dice标准差': std_dice_per_class.numpy(),
        'IoU分数': mean_iou_per_class.numpy(),
        'IoU标准差': std_iou_per_class.numpy()
    })

    # 按Dice系数排序
    results_df = results_df.sort_values('Dice系数', ascending=False)

    print("\n=== 前10个表现最好的解剖结构 ===")
    print(results_df.head(10).to_string(index=False))

    print("\n=== 前10个表现最差的解剖结构 ===")
    print(results_df.tail(10).to_string(index=False))

    # 保存完整结果
    results_path = os.path.join(OUTPUT_DIR, f'evaluation_results_{test_subject}.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n完整结果已保存到: {results_path}")


# ============================================================================
# 9. 可视化 - 评估指标分布
# ============================================================================

if model_loaded:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Dice系数直方图
    axes[0, 0].hist(mean_dice_per_class.numpy(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(overall_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_dice:.3f}')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Dice Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. IoU分数直方图
    axes[0, 1].hist(mean_iou_per_class.numpy(), bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(overall_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_iou:.3f}')
    axes[0, 1].set_xlabel('IoU Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('IoU Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Dice vs IoU散点图
    axes[1, 0].scatter(mean_dice_per_class.numpy(), mean_iou_per_class.numpy(), alpha=0.6)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    axes[1, 0].set_xlabel('Dice Score')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].set_title('Dice vs IoU Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. 成功率柱状图
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    success_rates = [calculate_success_rate(mean_dice_per_class, t) for t in thresholds]
    axes[1, 1].bar([str(t) for t in thresholds], [sr * 100 for sr in success_rates],
                   edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Dice Threshold')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Success Rate at Different Thresholds')
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    metrics_plot_path = os.path.join(OUTPUT_DIR, f'metrics_distribution_{test_subject}.png')
    plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n指标分布图已保存到: {metrics_plot_path}")
    plt.show()


# ============================================================================
# 10. 可视化 - 前20名表现最好/最差的结构
# ============================================================================

if model_loaded:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 最好的20个
    top_20 = results_df.head(20)
    y_pos = np.arange(len(top_20))
    axes[0].barh(y_pos, top_20['Dice系数'], xerr=top_20['Dice标准差'],
                 align='center', alpha=0.7, color='green', edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_20['解剖结构'], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Dice Score')
    axes[0].set_title('Top 20 Best Performing Structures')
    axes[0].grid(alpha=0.3, axis='x')

    # 最差的20个
    bottom_20 = results_df.tail(20)
    y_pos = np.arange(len(bottom_20))
    axes[1].barh(y_pos, bottom_20['Dice系数'], xerr=bottom_20['Dice标准差'],
                 align='center', alpha=0.7, color='red', edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(bottom_20['解剖结构'], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Dice Score')
    axes[1].set_title('Top 20 Worst Performing Structures')
    axes[1].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    ranking_plot_path = os.path.join(OUTPUT_DIR, f'structure_ranking_{test_subject}.png')
    plt.savefig(ranking_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n结构排名图已保存到: {ranking_plot_path}")
    plt.show()


# ============================================================================
# 11. 可视化 - 2D分割结果展示
# ============================================================================

if model_loaded and all_predictions:
    # 选择中间的一个batch进行可视化
    batch_idx = len(all_predictions) // 2
    sample_images = all_images[batch_idx]
    sample_preds = all_predictions[batch_idx]
    sample_targets = all_targets[batch_idx]

    # 选择几个有代表性的解剖结构
    structures_to_viz = ['liver', 'heart', 'kidney_left', 'kidney_right', 'lung_upper_lobe_left', 'aorta']
    channel_indices = []

    for struct in structures_to_viz:
        for filename, idx in label_map.items():
            if struct in filename.lower():
                channel_indices.append((idx, filename.replace('.nii.gz', '')))
                break

    # 选择第一个样本
    sample_idx = 0
    img = sample_images[sample_idx, 0].numpy()

    # 创建可视化
    n_structures = len(channel_indices)
    fig, axes = plt.subplots(n_structures, 3, figsize=(15, 5 * n_structures))

    if n_structures == 1:
        axes = axes.reshape(1, -1)

    for row, (ch_idx, struct_name) in enumerate(channel_indices):
        pred = (sample_preds[sample_idx, ch_idx] > 0.5).numpy()
        target = sample_targets[sample_idx, ch_idx].numpy()

        # 计算该结构的Dice
        dice = dice_coefficient(
            sample_preds[sample_idx:sample_idx+1, ch_idx:ch_idx+1],
            sample_targets[sample_idx:sample_idx+1, ch_idx:ch_idx+1]
        ).item()

        # CT图像
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title(f'{struct_name}\nOriginal CT')
        axes[row, 0].axis('off')

        # 真实掩码
        axes[row, 1].imshow(img, cmap='gray')
        axes[row, 1].imshow(target, cmap='Reds', alpha=0.5)
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')

        # 预测掩码
        axes[row, 2].imshow(img, cmap='gray')
        axes[row, 2].imshow(pred, cmap='Greens', alpha=0.5)
        axes[row, 2].set_title(f'Prediction\nDice: {dice:.3f}')
        axes[row, 2].axis('off')

    plt.tight_layout()
    seg_viz_path = os.path.join(OUTPUT_DIR, f'segmentation_visualization_{test_subject}.png')
    plt.savefig(seg_viz_path, dpi=150, bbox_inches='tight')
    print(f"\n分割可视化已保存到: {seg_viz_path}")
    plt.show()


# ============================================================================
# 12. 可视化 - 多切片对比
# ============================================================================

if model_loaded and all_predictions:
    # 选择一个特定的解剖结构（例如肝脏）
    target_structure = 'liver'
    target_channel = None

    for filename, idx in label_map.items():
        if target_structure in filename.lower():
            target_channel = idx
            target_name = filename.replace('.nii.gz', '')
            break

    if target_channel is not None:
        # 收集该结构的多个切片
        n_slices = min(16, len(all_images[0]))

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i in range(n_slices):
            batch_idx = i // len(all_images[0])
            sample_idx = i % len(all_images[0])

            if batch_idx >= len(all_images):
                break

            img = all_images[batch_idx][sample_idx, 0].numpy()
            pred = (all_predictions[batch_idx][sample_idx, target_channel] > 0.5).numpy()
            target = all_targets[batch_idx][sample_idx, target_channel].numpy()

            # 叠加显示
            axes[i].imshow(img, cmap='gray')
            axes[i].contour(target, colors='red', linewidths=2, alpha=0.7)
            axes[i].contour(pred, colors='green', linewidths=2, alpha=0.7, linestyles='dashed')
            axes[i].set_title(f'Slice {i+1}')
            axes[i].axis('off')

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Ground Truth'),
            Line2D([0], [0], color='green', linewidth=2, linestyle='dashed', label='Prediction')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12)
        fig.suptitle(f'Multi-slice Segmentation Comparison: {target_name}', fontsize=16, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        multi_slice_path = os.path.join(OUTPUT_DIR, f'multi_slice_{target_structure}_{test_subject}.png')
        plt.savefig(multi_slice_path, dpi=150, bbox_inches='tight')
        print(f"\n多切片对比图已保存到: {multi_slice_path}")
        plt.show()


# ============================================================================
# 13. 3D可视化 - 单个结构
# ============================================================================

def create_3d_mesh(volume, threshold=0.5, step_size=1):
    """
    使用Marching Cubes算法从3D体积创建网格

    Args:
        volume: 3D numpy数组
        threshold: 阈值
        step_size: 采样步长

    Returns:
        vertices, faces: 网格的顶点和面
    """
    try:
        verts, faces, normals, values = marching_cubes(
            volume,
            level=threshold,
            step_size=step_size,
            allow_degenerate=False
        )
        return verts, faces
    except Exception as e:
        print(f"创建网格失败: {e}")
        return None, None


if model_loaded:
    print("\n=== 创建3D可视化 ===")

    # 选择几个重要的解剖结构进行3D可视化
    structures_3d = ['liver', 'heart', 'kidney_left']

    for struct_name in structures_3d:
        # 找到对应的通道
        target_channel = None
        for filename, idx in label_map.items():
            if struct_name in filename.lower():
                target_channel = idx
                full_name = filename.replace('.nii.gz', '')
                break

        if target_channel is None:
            print(f"未找到结构: {struct_name}")
            continue

        # 加载真实的3D分割数据
        seg_path = os.path.join(DATA_DIR, test_subject, 'segmentations', f'{full_name}.nii.gz')
        if not os.path.exists(seg_path):
            print(f"文件不存在: {seg_path}")
            continue

        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()

        # 降采样以加快处理速度
        step_size = 2

        print(f"\n处理 {full_name}...")
        print(f"数据形状: {seg_data.shape}")

        # 创建网格
        verts, faces = create_3d_mesh(seg_data, threshold=0.5, step_size=step_size)

        if verts is not None and faces is not None:
            print(f"顶点数: {len(verts)}, 面数: {len(faces)}")

            # 创建3D可视化
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.7,
                    color='lightblue',
                    flatshading=True,
                    name=full_name
                )
            ])

            fig.update_layout(
                title=f'3D Visualization: {full_name}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                width=800,
                height=800
            )

            # 保存HTML
            html_path = os.path.join(OUTPUT_DIR, f'3d_{struct_name}_{test_subject}.html')
            fig.write_html(html_path)
            print(f"3D可视化已保存到: {html_path}")

            # 在notebook中显示
            fig.show()
        else:
            print(f"无法为 {full_name} 创建3D网格")


# ============================================================================
# 14. 3D可视化 - 多个结构组合
# ============================================================================

if model_loaded:
    print("\n=== 创建多结构组合3D可视化 ===")

    # 选择多个结构
    multi_structures = ['heart', 'liver', 'kidney_left', 'kidney_right']
    colors = ['red', 'brown', 'blue', 'lightblue']

    fig = go.Figure()

    for struct_name, color in zip(multi_structures, colors):
        # 找到对应的通道
        target_channel = None
        for filename, idx in label_map.items():
            if struct_name in filename.lower():
                full_name = filename.replace('.nii.gz', '')
                break

        # 加载3D分割数据
        seg_path = os.path.join(DATA_DIR, test_subject, 'segmentations', f'{full_name}.nii.gz')
        if not os.path.exists(seg_path):
            continue

        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()

        # 创建网格（使用较大的step_size以减少顶点数）
        verts, faces = create_3d_mesh(seg_data, threshold=0.5, step_size=3)

        if verts is not None and faces is not None:
            print(f"{full_name}: {len(verts)} vertices, {len(faces)} faces")

            fig.add_trace(go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.6,
                color=color,
                name=full_name,
                flatshading=True
            ))

    fig.update_layout(
        title=f'3D Multi-Structure Visualization: {test_subject}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=1000,
        showlegend=True
    )

    # 保存HTML
    multi_html_path = os.path.join(OUTPUT_DIR, f'3d_multi_structure_{test_subject}.html')
    fig.write_html(multi_html_path)
    print(f"\n多结构3D可视化已保存到: {multi_html_path}")

    # 显示
    fig.show()


# ============================================================================
# 15. 综合报告生成
# ============================================================================

if model_loaded:
    print("\n" + "="*60)
    print(" " * 15 + "3D医学图像分割检测综合报告")
    print("="*60)

    print(f"\n测试受试者: {test_subject}")
    print(f"测试切片数量: {len(test_dataset)}")
    print(f"解剖结构数量: {len(label_map)}")

    print("\n【总体性能指标】")
    print(f"  平均Dice系数: {overall_dice:.4f}")
    print(f"  平均IoU分数: {overall_iou:.4f}")

    print("\n【检测成功率】")
    print(f"  Dice > 0.50: {success_rate_50:>6.2%}  ({int(success_rate_50 * len(label_map))}/{len(label_map)}个结构)")
    print(f"  Dice > 0.70: {success_rate_70:>6.2%}  ({int(success_rate_70 * len(label_map))}/{len(label_map)}个结构)")
    print(f"  Dice > 0.80: {success_rate_80:>6.2%}  ({int(success_rate_80 * len(label_map))}/{len(label_map)}个结构)")

    print("\n【表现最好的5个结构】")
    for i, row in results_df.head(5).iterrows():
        print(f"  {row['解剖结构']:<30} Dice: {row['Dice系数']:.4f}  IoU: {row['IoU分数']:.4f}")

    print("\n【表现最差的5个结构】")
    for i, row in results_df.tail(5).iterrows():
        print(f"  {row['解剖结构']:<30} Dice: {row['Dice系数']:.4f}  IoU: {row['IoU分数']:.4f}")

    print("\n【统计摘要】")
    print(f"  Dice系数中位数: {results_df['Dice系数'].median():.4f}")
    print(f"  Dice系数标准差: {results_df['Dice系数'].std():.4f}")
    print(f"  最高Dice系数: {results_df['Dice系数'].max():.4f}")
    print(f"  最低Dice系数: {results_df['Dice系数'].min():.4f}")

    print("\n【输出文件】")
    output_files = [
        f'evaluation_results_{test_subject}.csv',
        f'metrics_distribution_{test_subject}.png',
        f'structure_ranking_{test_subject}.png',
        f'segmentation_visualization_{test_subject}.png',
    ]

    for filename in output_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")

    print("\n" + "="*60)
    print(" " * 20 + "报告生成完成！")
    print("="*60)


# ============================================================================
# 16. 交互式分析工具
# ============================================================================

if model_loaded:
    # 创建交互式Plotly图表用于探索所有结构的性能
    fig = go.Figure()

    # 添加散点图
    fig.add_trace(go.Scatter(
        x=results_df['Dice系数'],
        y=results_df['IoU分数'],
        mode='markers',
        marker=dict(
            size=8,
            color=results_df['Dice系数'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Dice Score')
        ),
        text=results_df['解剖结构'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Dice: %{x:.4f}<br>' +
                      'IoU: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title='Interactive Performance Analysis: Dice vs IoU',
        xaxis_title='Dice Score',
        yaxis_title='IoU Score',
        width=900,
        height=600,
        hovermode='closest'
    )

    # 保存交互式图表
    interactive_path = os.path.join(OUTPUT_DIR, f'interactive_analysis_{test_subject}.html')
    fig.write_html(interactive_path)
    print(f"\n交互式分析图表已保存到: {interactive_path}")

    fig.show()


# ============================================================================
# 总结
# ============================================================================

print("\n" + "="*60)
print("脚本执行完成")
print("\n本脚本实现了完整的3D医学图像分割检测与评估流程：")
print("1. 数据加载: 支持多受试者CT扫描和117个解剖结构的分割标注")
print("2. 模型推理: 使用训练好的U-Net模型进行2D切片分割")
print("3. 评估指标: 计算Dice系数、IoU、检测成功率等多项指标")
print("4. 2D可视化: 展示分割结果、指标分布、结构排名等")
print("5. 3D可视化: 生成交互式3D网格模型")
print("6. 综合报告: 生成详细的性能分析报告")
print("="*60)
