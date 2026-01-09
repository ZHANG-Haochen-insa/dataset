#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脂肪量分析脚本

本脚本用于对CT图像进行脂肪分割并计算皮下脂肪(SAT)和内脏脂肪(VAT)的面积和体积。

功能:
1. 加载训练好的U-Net模型或使用HU阈值方法
2. 对CT图像进行逐层分割
3. 计算SAT和VAT的面积（每层）和总体积
4. 生成分析报告和可视化结果

使用方法:
    python fat_analysis.py --ct_path /path/to/ct.nii.gz --output_dir /path/to/output

作者: hzhang02
日期: 2024-12-17
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage import morphology
from skimage.measure import label, regionprops
from scipy import ndimage
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from datetime import datetime
from tqdm import tqdm


# ============================================================================
# 1. 常量定义
# ============================================================================

# HU阈值定义
FAT_HU_MIN = -120  # 脂肪HU最小值
FAT_HU_MAX = -40   # 脂肪HU最大值
AIR_HU_MAX = -500  # 空气HU最大值
BODY_HU_MIN = -500 # 身体区域的最小HU值

# 脂肪名称
FAT_NAMES = {
    'SAT': {'zh': '皮下脂肪', 'fr': 'Tissu adipeux sous-cutané', 'en': 'Subcutaneous Adipose Tissue'},
    'VAT': {'zh': '内脏脂肪', 'fr': 'Tissu adipeux viscéral', 'en': 'Visceral Adipose Tissue'},
}

# 颜色定义
FAT_COLORS = {
    'SAT': '#FF6B6B',  # 红色
    'VAT': '#4ECDC4',  # 青色
}


# ============================================================================
# 2. 模型定义（与训练时保持一致）
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
# 3. HU阈值方法函数
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


def generate_fat_masks_threshold(slice_img: np.ndarray) -> tuple:
    """
    基于HU阈值生成SAT和VAT掩码

    Args:
        slice_img: CT切片 (H, W)，HU值

    Returns:
        sat_mask: 皮下脂肪掩码 (H, W)
        vat_mask: 内脏脂肪掩码 (H, W)
    """
    # 获取身体掩码
    body_mask = get_body_mask(slice_img)

    # 获取脂肪区域
    fat_mask = (slice_img >= FAT_HU_MIN) & (slice_img <= FAT_HU_MAX)
    fat_mask = fat_mask & body_mask.astype(bool)

    # 区分SAT和VAT
    subcutaneous_depth = 15
    eroded = morphology.binary_erosion(body_mask, morphology.disk(subcutaneous_depth))
    outer_region = body_mask.astype(bool) & ~eroded

    # SAT: 外围区域内的脂肪
    sat_mask = fat_mask & outer_region

    # VAT: 内部区域的脂肪
    vat_mask = fat_mask & eroded

    # 形态学后处理
    sat_mask = morphology.binary_opening(sat_mask, morphology.disk(2))
    sat_mask = morphology.binary_closing(sat_mask, morphology.disk(2))
    vat_mask = morphology.binary_opening(vat_mask, morphology.disk(2))
    vat_mask = morphology.binary_closing(vat_mask, morphology.disk(2))

    return sat_mask.astype(np.float32), vat_mask.astype(np.float32)


# ============================================================================
# 4. 分析类定义
# ============================================================================

class FatAnalyzer:
    """脂肪量分析器"""

    def __init__(self, model_path=None, device=None, use_model=True):
        """
        初始化分析器

        Args:
            model_path: 模型权重文件路径（可选）
            device: 计算设备 (cuda/cpu)
            use_model: 是否使用深度学习模型，False则使用HU阈值方法
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        self.use_model = use_model and model_path is not None
        self.target_shape = (256, 256)
        self.num_classes = 2  # SAT, VAT

        # 标签映射
        self.label_map = {'SAT': 0, 'VAT': 1}
        self.idx_to_label = {0: 'SAT', 1: 'VAT'}

        if self.use_model:
            # 加载模型
            if os.path.exists(model_path):
                self.model = UNet2D(in_ch=1, out_ch=2, features=[32, 64, 128, 256])
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print(f"模型加载成功: {model_path}")
            else:
                print(f"警告: 模型文件不存在: {model_path}")
                print("将使用HU阈值方法进行分析")
                self.use_model = False
        else:
            print("使用HU阈值方法进行分析")

    def preprocess_slice(self, slice_img):
        """
        预处理单个CT切片（用于模型输入）

        Args:
            slice_img: 原始CT切片 (H, W)

        Returns:
            预处理后的tensor (1, 1, H, W)
        """
        lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
        slice_normalized = np.clip(slice_img, lo, hi)
        if hi - lo > 0:
            slice_normalized = (slice_normalized - lo) / (hi - lo)
        else:
            slice_normalized = np.zeros_like(slice_normalized)

        slice_normalized = resize(slice_normalized, self.target_shape, order=1,
                                   preserve_range=True, anti_aliasing=True)

        img_t = torch.from_numpy(slice_normalized).unsqueeze(0).unsqueeze(0).float()
        return img_t

    def predict_slice(self, slice_img):
        """
        对单个切片进行预测

        Args:
            slice_img: 原始CT切片 (H, W)，HU值

        Returns:
            sat_mask: SAT掩码 (H, W)
            vat_mask: VAT掩码 (H, W)
        """
        if self.use_model:
            # 使用深度学习模型
            slice_tensor = self.preprocess_slice(slice_img)
            with torch.no_grad():
                slice_tensor = slice_tensor.to(self.device)
                pred = torch.sigmoid(self.model(slice_tensor))
                pred_bin = (pred > 0.5).float()
            pred_numpy = pred_bin[0].cpu().numpy()
            sat_mask = pred_numpy[0]
            vat_mask = pred_numpy[1]
        else:
            # 使用HU阈值方法
            sat_mask, vat_mask = generate_fat_masks_threshold(slice_img)
            # 调整大小以匹配目标形状
            sat_mask = resize(sat_mask, self.target_shape, order=0,
                             preserve_range=True, anti_aliasing=False)
            vat_mask = resize(vat_mask, self.target_shape, order=0,
                             preserve_range=True, anti_aliasing=False)
            sat_mask = (sat_mask > 0.5).astype(np.float32)
            vat_mask = (vat_mask > 0.5).astype(np.float32)

        return sat_mask, vat_mask

    def analyze_ct(self, ct_path, output_dir=None):
        """
        分析整个CT图像

        Args:
            ct_path: CT图像路径 (.nii.gz)
            output_dir: 输出目录（可选）

        Returns:
            分析结果字典
        """
        print(f"\n{'=' * 60}")
        print(f"开始分析: {ct_path}")
        print(f"分析方法: {'深度学习模型' if self.use_model else 'HU阈值方法'}")
        print(f"{'=' * 60}")

        # 加载CT图像
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata().astype(np.float32)
        header = ct_nii.header
        affine = ct_nii.affine

        # 获取体素尺寸
        voxel_dims = header.get_zooms()
        voxel_size_mm = float(voxel_dims[0]) * float(voxel_dims[1])
        slice_thickness = float(voxel_dims[2]) if len(voxel_dims) > 2 else 1.0
        voxel_volume_mm3 = voxel_size_mm * slice_thickness

        print(f"\nCT图像信息:")
        print(f"  形状: {ct_data.shape}")
        print(f"  体素尺寸: {voxel_dims[0]:.2f} x {voxel_dims[1]:.2f} x {slice_thickness:.2f} mm")
        print(f"  切片数量: {ct_data.shape[2]}")

        # 计算缩放因子
        original_shape = ct_data.shape[:2]
        scale_factor = (original_shape[0] / self.target_shape[0]) * (original_shape[1] / self.target_shape[1])

        # 初始化结果存储
        depth = ct_data.shape[2]
        all_sat_masks = np.zeros((*original_shape, depth), dtype=np.float32)
        all_vat_masks = np.zeros((*original_shape, depth), dtype=np.float32)
        sat_slice_areas = []
        vat_slice_areas = []

        # 逐层分割
        print(f"\n开始逐层分割...")
        for z in tqdm(range(depth), desc="处理切片"):
            slice_img = ct_data[:, :, z]

            # 预测
            sat_mask, vat_mask = self.predict_slice(slice_img)

            # 调整回原始尺寸
            sat_mask_resized = resize(sat_mask, original_shape, order=0,
                                      preserve_range=True, anti_aliasing=False)
            vat_mask_resized = resize(vat_mask, original_shape, order=0,
                                      preserve_range=True, anti_aliasing=False)

            all_sat_masks[:, :, z] = (sat_mask_resized > 0.5).astype(np.float32)
            all_vat_masks[:, :, z] = (vat_mask_resized > 0.5).astype(np.float32)

            # 计算面积
            sat_area = all_sat_masks[:, :, z].sum() * voxel_size_mm
            vat_area = all_vat_masks[:, :, z].sum() * voxel_size_mm
            sat_slice_areas.append(sat_area)
            vat_slice_areas.append(vat_area)

        # 计算总体积
        print(f"\n计算脂肪体积...")

        sat_total_pixels = all_sat_masks.sum()
        vat_total_pixels = all_vat_masks.sum()
        total_fat_pixels = sat_total_pixels + vat_total_pixels

        sat_volume_mm3 = sat_total_pixels * voxel_volume_mm3
        vat_volume_mm3 = vat_total_pixels * voxel_volume_mm3
        total_fat_volume_mm3 = sat_volume_mm3 + vat_volume_mm3

        sat_volume_cm3 = sat_volume_mm3 / 1000
        vat_volume_cm3 = vat_volume_mm3 / 1000
        total_fat_volume_cm3 = total_fat_volume_mm3 / 1000

        # 计算SAT/VAT比例
        if vat_volume_cm3 > 0:
            sat_vat_ratio = sat_volume_cm3 / vat_volume_cm3
        else:
            sat_vat_ratio = float('inf')

        # 找到脂肪出现的切片范围
        sat_slices = [i for i, a in enumerate(sat_slice_areas) if a > 0]
        vat_slices = [i for i, a in enumerate(vat_slice_areas) if a > 0]

        results = {
            'ct_path': ct_path,
            'ct_shape': list(ct_data.shape),
            'voxel_dims_mm': [float(v) for v in voxel_dims],
            'analysis_method': 'deep_learning' if self.use_model else 'hu_threshold',
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fat': {
                'SAT': {
                    'zh_name': FAT_NAMES['SAT']['zh'],
                    'en_name': FAT_NAMES['SAT']['en'],
                    'volume_mm3': float(sat_volume_mm3),
                    'volume_cm3': float(sat_volume_cm3),
                    'avg_area_mm2': float(np.mean([a for a in sat_slice_areas if a > 0]) if sat_slices else 0),
                    'max_area_mm2': float(max(sat_slice_areas) if sat_slice_areas else 0),
                    'slice_range': [min(sat_slices), max(sat_slices)] if sat_slices else [0, 0],
                    'num_slices': len(sat_slices),
                    'slice_areas_mm2': [float(a) for a in sat_slice_areas],
                },
                'VAT': {
                    'zh_name': FAT_NAMES['VAT']['zh'],
                    'en_name': FAT_NAMES['VAT']['en'],
                    'volume_mm3': float(vat_volume_mm3),
                    'volume_cm3': float(vat_volume_cm3),
                    'avg_area_mm2': float(np.mean([a for a in vat_slice_areas if a > 0]) if vat_slices else 0),
                    'max_area_mm2': float(max(vat_slice_areas) if vat_slice_areas else 0),
                    'slice_range': [min(vat_slices), max(vat_slices)] if vat_slices else [0, 0],
                    'num_slices': len(vat_slices),
                    'slice_areas_mm2': [float(a) for a in vat_slice_areas],
                },
            },
            'total_fat_volume_cm3': float(total_fat_volume_cm3),
            'sat_vat_ratio': float(sat_vat_ratio) if sat_vat_ratio != float('inf') else None,
            'sat_percentage': float(sat_volume_cm3 / total_fat_volume_cm3 * 100) if total_fat_volume_cm3 > 0 else 0,
            'vat_percentage': float(vat_volume_cm3 / total_fat_volume_cm3 * 100) if total_fat_volume_cm3 > 0 else 0,
        }

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._save_results(results, all_sat_masks, all_vat_masks, ct_data, ct_nii, output_dir)

        return results, all_sat_masks, all_vat_masks

    def _save_results(self, results, sat_masks, vat_masks, ct_data, ct_nii, output_dir):
        """保存分析结果"""

        print(f"\n保存结果到: {output_dir}")

        # 1. 保存JSON报告（不包含slice_areas）
        json_path = os.path.join(output_dir, 'fat_analysis_results.json')
        results_summary = {k: v for k, v in results.items() if k != 'fat'}
        results_summary['fat'] = {}
        for fat_type, fat_data in results['fat'].items():
            results_summary['fat'][fat_type] = {k: v for k, v in fat_data.items() if k != 'slice_areas_mm2'}

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"  JSON报告: {json_path}")

        # 2. 保存CSV报告
        csv_path = os.path.join(output_dir, 'fat_volumes.csv')
        df_data = []
        for fat_type, fat_data in results['fat'].items():
            df_data.append({
                '脂肪类型(中文)': fat_data['zh_name'],
                '脂肪类型(英文)': fat_data['en_name'],
                '体积(cm³)': f"{fat_data['volume_cm3']:.2f}",
                '体积(mm³)': f"{fat_data['volume_mm3']:.2f}",
                '平均面积(mm²)': f"{fat_data['avg_area_mm2']:.2f}",
                '最大面积(mm²)': f"{fat_data['max_area_mm2']:.2f}",
                '出现切片数': fat_data['num_slices'],
                '切片范围': f"{fat_data['slice_range'][0]}-{fat_data['slice_range'][1]}"
            })
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  CSV报告: {csv_path}")

        # 3. 保存分割结果为NIfTI
        # 合并SAT和VAT: 1=SAT, 2=VAT
        combined_mask = np.zeros(sat_masks.shape, dtype=np.int16)
        combined_mask[sat_masks > 0.5] = 1
        combined_mask[vat_masks > 0.5] = 2

        mask_nii = nib.Nifti1Image(combined_mask, ct_nii.affine, ct_nii.header)
        mask_path = os.path.join(output_dir, 'fat_segmentation_mask.nii.gz')
        nib.save(mask_nii, mask_path)
        print(f"  分割掩码: {mask_path}")

        # 4. 生成可视化图像
        self._generate_visualizations(results, sat_masks, vat_masks, ct_data, output_dir)

    def _generate_visualizations(self, results, sat_masks, vat_masks, ct_data, output_dir):
        """生成可视化图像"""

        # 图1: 脂肪体积条形图
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        fat_types = ['SAT', 'VAT']
        volumes = [results['fat']['SAT']['volume_cm3'], results['fat']['VAT']['volume_cm3']]
        colors = [FAT_COLORS['SAT'], FAT_COLORS['VAT']]
        labels = [FAT_NAMES['SAT']['zh'], FAT_NAMES['VAT']['zh']]

        bars = ax1.bar(labels, volumes, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('体积 (cm³)', fontsize=12)
        ax1.set_title('脂肪体积分析', fontsize=14)
        ax1.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, vol in zip(bars, volumes):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{vol:.1f} cm³', ha='center', fontsize=11)

        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'fat_volumes_chart.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  体积图表: fat_volumes_chart.png")

        # 图2: 饼图显示SAT/VAT比例
        fig2, ax2 = plt.subplots(figsize=(8, 8))

        sizes = [results['sat_percentage'], results['vat_percentage']]
        explode = (0.05, 0.05)
        ax2.pie(sizes, explode=explode, labels=[FAT_NAMES['SAT']['zh'], FAT_NAMES['VAT']['zh']],
                colors=[FAT_COLORS['SAT'], FAT_COLORS['VAT']],
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        ax2.set_title('脂肪组成比例', fontsize=14)

        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'fat_composition_pie.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  组成比例图: fat_composition_pie.png")

        # 图3: 选择中间切片进行可视化
        depth = ct_data.shape[2]
        sample_slices = [depth // 4, depth // 2, 3 * depth // 4]

        fig3, axes = plt.subplots(2, 3, figsize=(15, 10))

        for idx, z in enumerate(sample_slices):
            # 上排：原始CT
            axes[0, idx].imshow(ct_data[:, :, z].T, cmap='gray', origin='lower')
            axes[0, idx].set_title(f'CT 切片 {z}', fontsize=12)
            axes[0, idx].axis('off')

            # 下排：叠加分割结果
            axes[1, idx].imshow(ct_data[:, :, z].T, cmap='gray', origin='lower')

            # SAT (红色)
            sat_slice = sat_masks[:, :, z].T
            if sat_slice.sum() > 0:
                masked = np.ma.masked_where(sat_slice < 0.5, sat_slice)
                axes[1, idx].imshow(masked, alpha=0.4, cmap=ListedColormap([FAT_COLORS['SAT']]), origin='lower')

            # VAT (青色)
            vat_slice = vat_masks[:, :, z].T
            if vat_slice.sum() > 0:
                masked = np.ma.masked_where(vat_slice < 0.5, vat_slice)
                axes[1, idx].imshow(masked, alpha=0.4, cmap=ListedColormap([FAT_COLORS['VAT']]), origin='lower')

            axes[1, idx].set_title(f'SAT (红) + VAT (青) 切片 {z}', fontsize=12)
            axes[1, idx].axis('off')

        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'fat_segmentation_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  分割可视化: fat_segmentation_visualization.png")

        # 图4: 脂肪面积沿切片分布
        fig4, ax4 = plt.subplots(figsize=(14, 8))

        sat_areas = results['fat']['SAT']['slice_areas_mm2']
        vat_areas = results['fat']['VAT']['slice_areas_mm2']

        ax4.plot(range(len(sat_areas)), sat_areas, label=FAT_NAMES['SAT']['zh'],
                 color=FAT_COLORS['SAT'], linewidth=2)
        ax4.plot(range(len(vat_areas)), vat_areas, label=FAT_NAMES['VAT']['zh'],
                 color=FAT_COLORS['VAT'], linewidth=2)
        ax4.fill_between(range(len(sat_areas)), sat_areas, alpha=0.3, color=FAT_COLORS['SAT'])
        ax4.fill_between(range(len(vat_areas)), vat_areas, alpha=0.3, color=FAT_COLORS['VAT'])

        ax4.set_xlabel('切片编号', fontsize=12)
        ax4.set_ylabel('面积 (mm²)', fontsize=12)
        ax4.set_title('脂肪面积沿切片分布', fontsize=14)
        ax4.legend(loc='upper right', fontsize=11)
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        fig4.savefig(os.path.join(output_dir, 'fat_area_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"  面积分布图: fat_area_distribution.png")

    def print_report(self, results):
        """打印分析报告"""

        print(f"\n{'=' * 60}")
        print("脂肪量分析报告")
        print(f"{'=' * 60}")
        print(f"\n分析时间: {results['analysis_time']}")
        print(f"分析方法: {'深度学习模型' if results['analysis_method'] == 'deep_learning' else 'HU阈值方法'}")
        print(f"CT文件: {results['ct_path']}")
        print(f"图像尺寸: {results['ct_shape']}")
        print(f"体素尺寸: {results['voxel_dims_mm'][0]:.2f} x {results['voxel_dims_mm'][1]:.2f} x {results['voxel_dims_mm'][2]:.2f} mm")

        print(f"\n{'=' * 60}")
        print("脂肪体积统计")
        print(f"{'=' * 60}")
        print(f"\n{'脂肪类型':<20} {'体积(cm³)':>12} {'占比':>10} {'平均面积(mm²)':>15}")
        print("-" * 60)

        for fat_type, fat_data in results['fat'].items():
            percentage = results['sat_percentage'] if fat_type == 'SAT' else results['vat_percentage']
            print(f"{fat_data['zh_name']:<20} {fat_data['volume_cm3']:>12.2f} "
                  f"{percentage:>9.1f}% {fat_data['avg_area_mm2']:>15.2f}")

        print("-" * 60)
        print(f"{'总脂肪体积':<20} {results['total_fat_volume_cm3']:>12.2f} cm³")
        print(f"{'=' * 60}")

        # SAT/VAT比值分析
        print(f"\n{'=' * 60}")
        print("SAT/VAT 比值分析")
        print(f"{'=' * 60}")

        if results['sat_vat_ratio'] is not None:
            ratio = results['sat_vat_ratio']
            print(f"\nSAT/VAT 比值: {ratio:.2f}")

            # 健康评估
            if ratio > 1.5:
                assessment = "SAT占主导，内脏脂肪相对较低"
            elif ratio > 1.0:
                assessment = "SAT略高于VAT，分布较为均衡"
            elif ratio > 0.5:
                assessment = "VAT略高于SAT，需关注内脏脂肪"
            else:
                assessment = "VAT占主导，内脏脂肪偏高"

            print(f"评估: {assessment}")
        else:
            print("\n无法计算SAT/VAT比值（VAT为0）")

        print(f"\n{'=' * 60}")


# ============================================================================
# 5. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CT图像脂肪量分析')
    parser.add_argument('--ct_path', type=str, required=True, help='CT图像路径 (.nii.gz)')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--model_path', type=str,
                        default='/local/hzhang02/data/dataset/outputs_fat/best_fat_model.pth',
                        help='模型权重路径')
    parser.add_argument('--use_model', action='store_true', help='使用深度学习模型（默认使用HU阈值方法）')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (cuda/cpu)')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.ct_path):
        print(f"错误: CT文件不存在: {args.ct_path}")
        return

    # 设置输出目录
    if args.output_dir is None:
        ct_dir = os.path.dirname(args.ct_path)
        args.output_dir = os.path.join(ct_dir, 'fat_analysis')

    # 创建分析器
    analyzer = FatAnalyzer(
        model_path=args.model_path if args.use_model else None,
        device=args.device,
        use_model=args.use_model
    )

    # 执行分析
    results, sat_masks, vat_masks = analyzer.analyze_ct(args.ct_path, args.output_dir)

    # 打印报告
    analyzer.print_report(results)

    print(f"\n分析完成！结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
