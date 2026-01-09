#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版肌肉量分析脚本

本脚本用于对CT图像进行肌肉分割并计算各类肌肉的面积和体积。
在原有10种特定肌肉分割基础上，额外计算环腹部肌肉总量（腹壁肌肉）。

功能:
1. 使用训练好的U-Net模型分割10种特定肌肉
2. 使用HU阈值方法检测环腹部肌肉总量（腹壁肌肉）
3. 计算每种肌肉的面积（每层）和总体积
4. 生成分析报告和可视化结果

环腹部肌肉包括:
- 腹直肌 (Rectus Abdominis)
- 腹外斜肌 (External Oblique)
- 腹内斜肌 (Internal Oblique)
- 腹横肌 (Transversus Abdominis)
- 其他腹壁肌肉

HU值范围:
- 肌肉组织: -29 ~ 150 HU (典型范围: 10 ~ 40 HU)
- 脂肪组织: -190 ~ -30 HU

使用方法:
    python muscle_analysis_enhanced.py --ct_path /path/to/ct.nii.gz --output_dir /path/to/output

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
MUSCLE_HU_MIN = -29    # 肌肉HU最小值
MUSCLE_HU_MAX = 150    # 肌肉HU最大值
FAT_HU_MIN = -190      # 脂肪HU最小值
FAT_HU_MAX = -30       # 脂肪HU最大值
BODY_HU_MIN = -500     # 身体区域的最小HU值

# 原有10种肌肉名称
MUSCLE_NAMES = {
    'autochthon_left': {'zh': '左侧背部肌群', 'fr': 'Muscles autochtones gauches', 'en': 'Left Autochthon'},
    'autochthon_right': {'zh': '右侧背部肌群', 'fr': 'Muscles autochtones droits', 'en': 'Right Autochthon'},
    'gluteus_maximus_left': {'zh': '左臀大肌', 'fr': 'Grand fessier gauche', 'en': 'Left Gluteus Maximus'},
    'gluteus_maximus_right': {'zh': '右臀大肌', 'fr': 'Grand fessier droit', 'en': 'Right Gluteus Maximus'},
    'gluteus_medius_left': {'zh': '左臀中肌', 'fr': 'Moyen fessier gauche', 'en': 'Left Gluteus Medius'},
    'gluteus_medius_right': {'zh': '右臀中肌', 'fr': 'Moyen fessier droit', 'en': 'Right Gluteus Medius'},
    'gluteus_minimus_left': {'zh': '左臀小肌', 'fr': 'Petit fessier gauche', 'en': 'Left Gluteus Minimus'},
    'gluteus_minimus_right': {'zh': '右臀小肌', 'fr': 'Petit fessier droit', 'en': 'Right Gluteus Minimus'},
    'iliopsoas_left': {'zh': '左髂腰肌', 'fr': 'Ilio-psoas gauche', 'en': 'Left Iliopsoas'},
    'iliopsoas_right': {'zh': '右髂腰肌', 'fr': 'Ilio-psoas droit', 'en': 'Right Iliopsoas'},
}

# 腹壁肌肉名称（环腹部肌肉）
ABDOMINAL_WALL_MUSCLE = {
    'abdominal_wall': {
        'zh': '环腹部肌肉总量',
        'fr': 'Muscles de la paroi abdominale',
        'en': 'Abdominal Wall Muscles',
        'description_zh': '包括腹直肌、腹外斜肌、腹内斜肌、腹横肌等',
        'description_en': 'Including Rectus Abdominis, External/Internal Oblique, Transversus Abdominis'
    }
}

# 肌肉颜色
MUSCLE_COLORS = [
    '#FF6B6B',  # 左侧背部肌群 - 红色
    '#4ECDC4',  # 右侧背部肌群 - 青色
    '#45B7D1',  # 左臀大肌 - 蓝色
    '#96CEB4',  # 右臀大肌 - 绿色
    '#FFEAA7',  # 左臀中肌 - 黄色
    '#DDA0DD',  # 右臀中肌 - 紫色
    '#98D8C8',  # 左臀小肌 - 薄荷绿
    '#F7DC6F',  # 右臀小肌 - 金色
    '#BB8FCE',  # 左髂腰肌 - 淡紫色
    '#85C1E9',  # 右髂腰肌 - 淡蓝色
    '#E74C3C',  # 环腹部肌肉 - 深红色
]


# ============================================================================
# 2. 模型定义（与训练时保持一致）
# ============================================================================

class DoubleConv(nn.Module):
    """双卷积块"""

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
# 3. 环腹部肌肉检测函数（基于HU阈值）
# ============================================================================

def get_body_mask(slice_img: np.ndarray, threshold: float = -500) -> np.ndarray:
    """获取身体区域掩码"""
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


def detect_abdominal_wall_muscle(slice_img: np.ndarray,
                                  segmented_muscles: np.ndarray = None) -> np.ndarray:
    """
    检测环腹部肌肉（腹壁肌肉）

    方法:
    1. 基于HU阈值检测所有肌肉组织
    2. 定位腹壁区域（皮下脂肪内侧，内脏区域外侧）
    3. 排除已分割的特定肌肉

    Args:
        slice_img: CT切片 (H, W)，HU值
        segmented_muscles: 已分割的肌肉掩码 (H, W)，可选

    Returns:
        abdominal_wall_mask: 腹壁肌肉掩码 (H, W)
    """
    # 1. 获取身体掩码
    body_mask = get_body_mask(slice_img)

    # 2. 检测所有肌肉组织（基于HU阈值）
    muscle_mask = (slice_img >= MUSCLE_HU_MIN) & (slice_img <= MUSCLE_HU_MAX)
    muscle_mask = muscle_mask & body_mask.astype(bool)

    # 3. 定位腹壁区域
    # 腹壁位于：皮下脂肪内侧 到 腹腔边界外侧
    # 通过形态学操作定义这个区域

    # 皮下区域深度（从皮肤向内）
    subcutaneous_depth = 15

    # 腹腔区域（进一步腐蚀得到核心区域）
    abdominal_cavity_depth = 40

    # 外围边界（皮下脂肪区域）
    outer_boundary = morphology.binary_erosion(body_mask, morphology.disk(subcutaneous_depth))

    # 腹腔核心区域
    inner_core = morphology.binary_erosion(body_mask, morphology.disk(abdominal_cavity_depth))

    # 腹壁区域 = 外围边界 - 腹腔核心
    abdominal_wall_region = outer_boundary.astype(bool) & ~inner_core

    # 4. 腹壁肌肉 = 腹壁区域内的肌肉组织
    abdominal_wall_muscle = muscle_mask & abdominal_wall_region

    # 5. 排除已分割的特定肌肉
    if segmented_muscles is not None and segmented_muscles.sum() > 0:
        # 确保尺寸匹配
        if segmented_muscles.shape != abdominal_wall_muscle.shape:
            segmented_muscles = resize(segmented_muscles, abdominal_wall_muscle.shape,
                                       order=0, preserve_range=True, anti_aliasing=False)
        abdominal_wall_muscle = abdominal_wall_muscle & ~(segmented_muscles > 0.5)

    # 6. 形态学后处理
    abdominal_wall_muscle = morphology.binary_opening(abdominal_wall_muscle, morphology.disk(2))
    abdominal_wall_muscle = morphology.binary_closing(abdominal_wall_muscle, morphology.disk(3))

    # 移除小的噪声区域
    labeled = label(abdominal_wall_muscle)
    if labeled.max() > 0:
        for region in regionprops(labeled):
            if region.area < 50:  # 移除面积小于50像素的区域
                abdominal_wall_muscle[labeled == region.label] = 0

    return abdominal_wall_muscle.astype(np.float32)


def detect_total_skeletal_muscle(slice_img: np.ndarray) -> np.ndarray:
    """
    检测切片中的所有骨骼肌（用于计算总肌肉量）

    Args:
        slice_img: CT切片 (H, W)，HU值

    Returns:
        total_muscle_mask: 总肌肉掩码 (H, W)
    """
    body_mask = get_body_mask(slice_img)

    # 检测所有肌肉组织
    muscle_mask = (slice_img >= MUSCLE_HU_MIN) & (slice_img <= MUSCLE_HU_MAX)
    muscle_mask = muscle_mask & body_mask.astype(bool)

    # 形态学后处理
    muscle_mask = morphology.binary_opening(muscle_mask, morphology.disk(2))
    muscle_mask = morphology.binary_closing(muscle_mask, morphology.disk(2))

    return muscle_mask.astype(np.float32)


# ============================================================================
# 4. 分析类定义
# ============================================================================

class EnhancedMuscleAnalyzer:
    """增强版肌肉量分析器"""

    def __init__(self, model_path, label_map_path, device=None):
        """
        初始化分析器

        Args:
            model_path: 模型权重文件路径
            label_map_path: 标签映射文件路径
            device: 计算设备 (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        # 加载标签映射
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.num_classes = len(self.label_map)
        print(f"加载标签映射: {self.num_classes} 种特定肌肉")

        # 创建反向映射
        self.idx_to_label = {v: k.replace('.nii.gz', '') for k, v in self.label_map.items()}

        # 加载模型
        self.model = UNet2D(in_ch=1, out_ch=self.num_classes, features=[32, 64, 128, 256])
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载成功: {model_path}")

        self.target_shape = (256, 256)

    def preprocess_slice(self, slice_img):
        """预处理单个CT切片"""
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

    def predict_slice(self, slice_tensor):
        """对单个切片进行模型预测"""
        with torch.no_grad():
            slice_tensor = slice_tensor.to(self.device)
            pred = torch.sigmoid(self.model(slice_tensor))
            pred_bin = (pred > 0.5).float()
        return pred_bin[0].cpu().numpy()

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
        print(f"开始增强版肌肉分析: {ct_path}")
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

        original_shape = ct_data.shape[:2]
        depth = ct_data.shape[2]

        # 初始化存储
        # 10种特定肌肉的预测结果
        all_predictions = np.zeros((self.num_classes, *original_shape, depth), dtype=np.float32)
        slice_areas = {i: [] for i in range(self.num_classes)}

        # 环腹部肌肉
        abdominal_wall_masks = np.zeros((*original_shape, depth), dtype=np.float32)
        abdominal_wall_areas = []

        # 总肌肉量
        total_muscle_masks = np.zeros((*original_shape, depth), dtype=np.float32)
        total_muscle_areas = []

        # 逐层分割
        print(f"\n开始逐层分割...")
        for z in tqdm(range(depth), desc="处理切片"):
            slice_img = ct_data[:, :, z]

            # 1. 使用U-Net模型预测10种特定肌肉
            slice_tensor = self.preprocess_slice(slice_img)
            pred_mask = self.predict_slice(slice_tensor)

            # 调整回原始尺寸并存储
            segmented_muscles_combined = np.zeros(original_shape, dtype=np.float32)
            for c in range(self.num_classes):
                pred_resized = resize(pred_mask[c], original_shape, order=0,
                                      preserve_range=True, anti_aliasing=False)
                all_predictions[c, :, :, z] = (pred_resized > 0.5).astype(np.float32)
                segmented_muscles_combined += all_predictions[c, :, :, z]

                # 计算该层该肌肉的面积
                pixel_count = all_predictions[c, :, :, z].sum()
                area_mm2 = pixel_count * voxel_size_mm
                slice_areas[c].append(area_mm2)

            # 2. 检测环腹部肌肉（排除已分割的特定肌肉）
            abdominal_wall = detect_abdominal_wall_muscle(slice_img, segmented_muscles_combined)
            abdominal_wall_masks[:, :, z] = abdominal_wall
            abdominal_wall_area = abdominal_wall.sum() * voxel_size_mm
            abdominal_wall_areas.append(abdominal_wall_area)

            # 3. 检测总肌肉量
            total_muscle = detect_total_skeletal_muscle(slice_img)
            total_muscle_masks[:, :, z] = total_muscle
            total_muscle_area = total_muscle.sum() * voxel_size_mm
            total_muscle_areas.append(total_muscle_area)

        # 计算体积
        print(f"\n计算肌肉体积...")
        results = {
            'ct_path': ct_path,
            'ct_shape': list(ct_data.shape),
            'voxel_dims_mm': [float(v) for v in voxel_dims],
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'specific_muscles': {},
            'abdominal_wall_muscle': {},
            'total_muscle': {},
        }

        # 10种特定肌肉的体积计算
        total_specific_volume = 0
        for c in range(self.num_classes):
            muscle_name = self.idx_to_label[c]
            muscle_zh = MUSCLE_NAMES[muscle_name]['zh']

            total_pixels = all_predictions[c].sum()
            volume_mm3 = total_pixels * voxel_volume_mm3
            volume_cm3 = volume_mm3 / 1000

            non_zero_areas = [a for a in slice_areas[c] if a > 0]
            avg_area_mm2 = np.mean(non_zero_areas) if non_zero_areas else 0
            max_area_mm2 = max(slice_areas[c]) if slice_areas[c] else 0

            slices_with_muscle = [i for i, a in enumerate(slice_areas[c]) if a > 0]
            slice_range = [min(slices_with_muscle), max(slices_with_muscle)] if slices_with_muscle else [0, 0]

            results['specific_muscles'][muscle_name] = {
                'zh_name': muscle_zh,
                'en_name': MUSCLE_NAMES[muscle_name]['en'],
                'volume_mm3': float(volume_mm3),
                'volume_cm3': float(volume_cm3),
                'avg_area_mm2': float(avg_area_mm2),
                'max_area_mm2': float(max_area_mm2),
                'slice_range': slice_range,
                'num_slices': int(len(slices_with_muscle)),
                'slice_areas_mm2': [float(a) for a in slice_areas[c]]
            }
            total_specific_volume += volume_cm3

        results['total_specific_muscles_volume_cm3'] = float(total_specific_volume)

        # 环腹部肌肉体积
        abdominal_wall_pixels = abdominal_wall_masks.sum()
        abdominal_wall_volume_mm3 = abdominal_wall_pixels * voxel_volume_mm3
        abdominal_wall_volume_cm3 = abdominal_wall_volume_mm3 / 1000

        non_zero_aw_areas = [a for a in abdominal_wall_areas if a > 0]
        aw_slices = [i for i, a in enumerate(abdominal_wall_areas) if a > 0]

        results['abdominal_wall_muscle'] = {
            'zh_name': ABDOMINAL_WALL_MUSCLE['abdominal_wall']['zh'],
            'en_name': ABDOMINAL_WALL_MUSCLE['abdominal_wall']['en'],
            'description_zh': ABDOMINAL_WALL_MUSCLE['abdominal_wall']['description_zh'],
            'description_en': ABDOMINAL_WALL_MUSCLE['abdominal_wall']['description_en'],
            'volume_mm3': float(abdominal_wall_volume_mm3),
            'volume_cm3': float(abdominal_wall_volume_cm3),
            'avg_area_mm2': float(np.mean(non_zero_aw_areas) if non_zero_aw_areas else 0),
            'max_area_mm2': float(max(abdominal_wall_areas) if abdominal_wall_areas else 0),
            'slice_range': [min(aw_slices), max(aw_slices)] if aw_slices else [0, 0],
            'num_slices': len(aw_slices),
            'slice_areas_mm2': [float(a) for a in abdominal_wall_areas]
        }

        # 总肌肉量
        total_muscle_pixels = total_muscle_masks.sum()
        total_muscle_volume_mm3 = total_muscle_pixels * voxel_volume_mm3
        total_muscle_volume_cm3 = total_muscle_volume_mm3 / 1000

        non_zero_tm_areas = [a for a in total_muscle_areas if a > 0]
        tm_slices = [i for i, a in enumerate(total_muscle_areas) if a > 0]

        results['total_muscle'] = {
            'zh_name': '总骨骼肌量',
            'en_name': 'Total Skeletal Muscle',
            'volume_mm3': float(total_muscle_volume_mm3),
            'volume_cm3': float(total_muscle_volume_cm3),
            'avg_area_mm2': float(np.mean(non_zero_tm_areas) if non_zero_tm_areas else 0),
            'max_area_mm2': float(max(total_muscle_areas) if total_muscle_areas else 0),
            'slice_range': [min(tm_slices), max(tm_slices)] if tm_slices else [0, 0],
            'num_slices': len(tm_slices),
            'slice_areas_mm2': [float(a) for a in total_muscle_areas]
        }

        # 计算各部分占比
        if total_muscle_volume_cm3 > 0:
            results['specific_muscles_percentage'] = float(total_specific_volume / total_muscle_volume_cm3 * 100)
            results['abdominal_wall_percentage'] = float(abdominal_wall_volume_cm3 / total_muscle_volume_cm3 * 100)
        else:
            results['specific_muscles_percentage'] = 0
            results['abdominal_wall_percentage'] = 0

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._save_results(results, all_predictions, abdominal_wall_masks,
                               total_muscle_masks, ct_data, ct_nii, output_dir)

        return results, all_predictions, abdominal_wall_masks, total_muscle_masks

    def _save_results(self, results, predictions, abdominal_wall_masks,
                      total_muscle_masks, ct_data, ct_nii, output_dir):
        """保存分析结果"""

        print(f"\n保存结果到: {output_dir}")

        # 1. 保存JSON报告
        json_path = os.path.join(output_dir, 'muscle_analysis_enhanced_results.json')
        results_summary = {k: v for k, v in results.items()
                          if k not in ['specific_muscles', 'abdominal_wall_muscle', 'total_muscle']}

        results_summary['specific_muscles'] = {}
        for muscle_name, muscle_data in results['specific_muscles'].items():
            results_summary['specific_muscles'][muscle_name] = {
                k: v for k, v in muscle_data.items() if k != 'slice_areas_mm2'
            }

        results_summary['abdominal_wall_muscle'] = {
            k: v for k, v in results['abdominal_wall_muscle'].items() if k != 'slice_areas_mm2'
        }

        results_summary['total_muscle'] = {
            k: v for k, v in results['total_muscle'].items() if k != 'slice_areas_mm2'
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"  JSON报告: {json_path}")

        # 2. 保存CSV报告
        csv_path = os.path.join(output_dir, 'muscle_volumes_enhanced.csv')
        df_data = []

        # 添加10种特定肌肉
        for muscle_name, muscle_data in results['specific_muscles'].items():
            df_data.append({
                '分类': '特定肌肉',
                '肌肉名称(中文)': muscle_data['zh_name'],
                '肌肉名称(英文)': muscle_data['en_name'],
                '体积(cm³)': f"{muscle_data['volume_cm3']:.2f}",
                '体积(mm³)': f"{muscle_data['volume_mm3']:.2f}",
                '平均面积(mm²)': f"{muscle_data['avg_area_mm2']:.2f}",
                '最大面积(mm²)': f"{muscle_data['max_area_mm2']:.2f}",
                '出现切片数': muscle_data['num_slices'],
            })

        # 添加环腹部肌肉
        aw = results['abdominal_wall_muscle']
        df_data.append({
            '分类': '环腹部肌肉',
            '肌肉名称(中文)': aw['zh_name'],
            '肌肉名称(英文)': aw['en_name'],
            '体积(cm³)': f"{aw['volume_cm3']:.2f}",
            '体积(mm³)': f"{aw['volume_mm3']:.2f}",
            '平均面积(mm²)': f"{aw['avg_area_mm2']:.2f}",
            '最大面积(mm²)': f"{aw['max_area_mm2']:.2f}",
            '出现切片数': aw['num_slices'],
        })

        # 添加总肌肉量
        tm = results['total_muscle']
        df_data.append({
            '分类': '总计',
            '肌肉名称(中文)': tm['zh_name'],
            '肌肉名称(英文)': tm['en_name'],
            '体积(cm³)': f"{tm['volume_cm3']:.2f}",
            '体积(mm³)': f"{tm['volume_mm3']:.2f}",
            '平均面积(mm²)': f"{tm['avg_area_mm2']:.2f}",
            '最大面积(mm²)': f"{tm['max_area_mm2']:.2f}",
            '出现切片数': tm['num_slices'],
        })

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  CSV报告: {csv_path}")

        # 3. 保存分割结果为NIfTI
        # 合并所有肌肉掩码
        # 1-10: 特定肌肉, 11: 环腹部肌肉
        combined_mask = np.zeros(predictions.shape[1:], dtype=np.int16)
        for c in range(self.num_classes):
            combined_mask[predictions[c] > 0.5] = c + 1
        combined_mask[abdominal_wall_masks > 0.5] = 11  # 环腹部肌肉

        mask_nii = nib.Nifti1Image(combined_mask, ct_nii.affine, ct_nii.header)
        mask_path = os.path.join(output_dir, 'muscle_segmentation_enhanced.nii.gz')
        nib.save(mask_nii, mask_path)
        print(f"  分割掩码: {mask_path}")

        # 4. 生成可视化
        self._generate_visualizations(results, predictions, abdominal_wall_masks,
                                       total_muscle_masks, ct_data, output_dir)

    def _generate_visualizations(self, results, predictions, abdominal_wall_masks,
                                  total_muscle_masks, ct_data, output_dir):
        """生成可视化图像"""

        # 图1: 肌肉体积条形图
        fig1, ax1 = plt.subplots(figsize=(14, 10))

        muscles = []
        volumes = []
        colors = []

        # 添加10种特定肌肉
        for i, (muscle_name, muscle_data) in enumerate(results['specific_muscles'].items()):
            muscles.append(muscle_data['zh_name'])
            volumes.append(muscle_data['volume_cm3'])
            colors.append(MUSCLE_COLORS[i % len(MUSCLE_COLORS)])

        # 添加环腹部肌肉
        muscles.append(results['abdominal_wall_muscle']['zh_name'])
        volumes.append(results['abdominal_wall_muscle']['volume_cm3'])
        colors.append(MUSCLE_COLORS[10])

        bars = ax1.barh(muscles, volumes, color=colors)
        ax1.set_xlabel('体积 (cm³)', fontsize=12)
        ax1.set_title('肌肉体积分析（增强版）', fontsize=14)
        ax1.grid(axis='x', alpha=0.3)

        for bar, vol in zip(bars, volumes):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{vol:.1f}', va='center', fontsize=10)

        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'muscle_volumes_enhanced_chart.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  体积图表: muscle_volumes_enhanced_chart.png")

        # 图2: 肌肉组成饼图
        fig2, ax2 = plt.subplots(figsize=(10, 10))

        # 分组：特定肌肉 vs 环腹部肌肉 vs 其他
        specific_vol = results['total_specific_muscles_volume_cm3']
        abdominal_vol = results['abdominal_wall_muscle']['volume_cm3']
        total_vol = results['total_muscle']['volume_cm3']
        other_vol = max(0, total_vol - specific_vol - abdominal_vol)

        sizes = [specific_vol, abdominal_vol, other_vol]
        labels = ['特定肌肉\n(深度学习分割)', '环腹部肌肉\n(腹壁肌群)', '其他肌肉']
        colors_pie = ['#3498db', '#e74c3c', '#95a5a6']
        explode = (0.05, 0.05, 0.02)

        ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        ax2.set_title('肌肉组成分析', fontsize=14)

        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'muscle_composition_pie.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  组成比例图: muscle_composition_pie.png")

        # 图3: 切片可视化
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

            # 显示特定肌肉
            for c in range(self.num_classes):
                mask = predictions[c, :, :, z].T
                if mask.sum() > 0:
                    masked = np.ma.masked_where(mask < 0.5, mask)
                    axes[1, idx].imshow(masked, alpha=0.3,
                                         cmap=ListedColormap([MUSCLE_COLORS[c]]), origin='lower')

            # 显示环腹部肌肉（红色轮廓）
            aw_mask = abdominal_wall_masks[:, :, z].T
            if aw_mask.sum() > 0:
                axes[1, idx].contour(aw_mask, levels=[0.5], colors=['red'], linewidths=2)
                masked = np.ma.masked_where(aw_mask < 0.5, aw_mask)
                axes[1, idx].imshow(masked, alpha=0.3,
                                     cmap=ListedColormap(['#E74C3C']), origin='lower')

            axes[1, idx].set_title(f'分割结果 切片 {z}\n(红色轮廓=环腹部肌肉)', fontsize=11)
            axes[1, idx].axis('off')

        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'segmentation_visualization_enhanced.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  分割可视化: segmentation_visualization_enhanced.png")

        # 图4: 面积分布图
        fig4, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 上图：特定肌肉面积分布
        for c in range(self.num_classes):
            muscle_name = self.idx_to_label[c]
            muscle_zh = MUSCLE_NAMES[muscle_name]['zh']
            areas = results['specific_muscles'][muscle_name]['slice_areas_mm2']
            axes[0].plot(range(len(areas)), areas, label=muscle_zh,
                         color=MUSCLE_COLORS[c], linewidth=1.5, alpha=0.8)

        axes[0].set_xlabel('切片编号', fontsize=12)
        axes[0].set_ylabel('面积 (mm²)', fontsize=12)
        axes[0].set_title('特定肌肉面积沿切片分布', fontsize=14)
        axes[0].legend(loc='upper right', fontsize=8, ncol=2)
        axes[0].grid(alpha=0.3)

        # 下图：环腹部肌肉和总肌肉面积分布
        aw_areas = results['abdominal_wall_muscle']['slice_areas_mm2']
        tm_areas = results['total_muscle']['slice_areas_mm2']

        axes[1].fill_between(range(len(tm_areas)), tm_areas, alpha=0.3, color='gray', label='总骨骼肌')
        axes[1].plot(range(len(tm_areas)), tm_areas, color='gray', linewidth=2)
        axes[1].fill_between(range(len(aw_areas)), aw_areas, alpha=0.5, color='#E74C3C', label='环腹部肌肉')
        axes[1].plot(range(len(aw_areas)), aw_areas, color='#E74C3C', linewidth=2)

        axes[1].set_xlabel('切片编号', fontsize=12)
        axes[1].set_ylabel('面积 (mm²)', fontsize=12)
        axes[1].set_title('环腹部肌肉与总肌肉面积沿切片分布', fontsize=14)
        axes[1].legend(loc='upper right', fontsize=11)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        fig4.savefig(os.path.join(output_dir, 'muscle_area_distribution_enhanced.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"  面积分布图: muscle_area_distribution_enhanced.png")

    def print_report(self, results):
        """打印分析报告"""

        print(f"\n{'=' * 70}")
        print("增强版肌肉量分析报告")
        print(f"{'=' * 70}")
        print(f"\n分析时间: {results['analysis_time']}")
        print(f"CT文件: {results['ct_path']}")
        print(f"图像尺寸: {results['ct_shape']}")
        print(f"体素尺寸: {results['voxel_dims_mm'][0]:.2f} x {results['voxel_dims_mm'][1]:.2f} x {results['voxel_dims_mm'][2]:.2f} mm")

        # 特定肌肉统计
        print(f"\n{'=' * 70}")
        print("一、特定肌肉体积统计（深度学习分割）")
        print(f"{'=' * 70}")
        print(f"\n{'肌肉名称':<20} {'体积(cm³)':>12} {'平均面积(mm²)':>15} {'切片数':>8}")
        print("-" * 60)

        for muscle_name, muscle_data in results['specific_muscles'].items():
            print(f"{muscle_data['zh_name']:<20} {muscle_data['volume_cm3']:>12.2f} "
                  f"{muscle_data['avg_area_mm2']:>15.2f} {muscle_data['num_slices']:>8}")

        print("-" * 60)
        print(f"{'特定肌肉总量':<20} {results['total_specific_muscles_volume_cm3']:>12.2f} cm³")

        # 环腹部肌肉统计
        print(f"\n{'=' * 70}")
        print("二、环腹部肌肉统计（HU阈值检测）")
        print(f"{'=' * 70}")
        aw = results['abdominal_wall_muscle']
        print(f"\n{aw['zh_name']}")
        print(f"  {aw['description_zh']}")
        print(f"\n  体积: {aw['volume_cm3']:.2f} cm³")
        print(f"  平均面积: {aw['avg_area_mm2']:.2f} mm²")
        print(f"  最大面积: {aw['max_area_mm2']:.2f} mm²")
        print(f"  出现切片数: {aw['num_slices']}")
        print(f"  切片范围: {aw['slice_range'][0]} - {aw['slice_range'][1]}")

        # 总肌肉量统计
        print(f"\n{'=' * 70}")
        print("三、总骨骼肌统计")
        print(f"{'=' * 70}")
        tm = results['total_muscle']
        print(f"\n  总骨骼肌体积: {tm['volume_cm3']:.2f} cm³")
        print(f"  特定肌肉占比: {results['specific_muscles_percentage']:.1f}%")
        print(f"  环腹部肌肉占比: {results['abdominal_wall_percentage']:.1f}%")

        # 左右对比
        print(f"\n{'=' * 70}")
        print("四、左右侧肌肉对比")
        print(f"{'=' * 70}")

        muscle_pairs = [
            ('autochthon_left', 'autochthon_right', '背部肌群'),
            ('gluteus_maximus_left', 'gluteus_maximus_right', '臀大肌'),
            ('gluteus_medius_left', 'gluteus_medius_right', '臀中肌'),
            ('gluteus_minimus_left', 'gluteus_minimus_right', '臀小肌'),
            ('iliopsoas_left', 'iliopsoas_right', '髂腰肌'),
        ]

        print(f"\n{'肌肉':<12} {'左侧(cm³)':>12} {'右侧(cm³)':>12} {'差异':>12} {'差异比例':>12}")
        print("-" * 60)

        for left, right, name in muscle_pairs:
            left_vol = results['specific_muscles'][left]['volume_cm3']
            right_vol = results['specific_muscles'][right]['volume_cm3']
            diff = left_vol - right_vol
            avg = (left_vol + right_vol) / 2
            diff_pct = (diff / avg * 100) if avg > 0 else 0

            print(f"{name:<12} {left_vol:>12.2f} {right_vol:>12.2f} {diff:>+12.2f} {diff_pct:>+11.1f}%")

        print(f"{'=' * 70}")


# ============================================================================
# 5. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='增强版CT图像肌肉量分析')
    parser.add_argument('--ct_path', type=str, required=True, help='CT图像路径 (.nii.gz)')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--model_path', type=str,
                        default='/local/hzhang02/data/dataset/outputs/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--label_map_path', type=str,
                        default='/local/hzhang02/data/dataset/outputs/label_map.json',
                        help='标签映射文件路径')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (cuda/cpu)')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.ct_path):
        print(f"错误: CT文件不存在: {args.ct_path}")
        return

    # 设置输出目录
    if args.output_dir is None:
        ct_dir = os.path.dirname(args.ct_path)
        args.output_dir = os.path.join(ct_dir, 'muscle_analysis_enhanced')

    # 创建分析器
    analyzer = EnhancedMuscleAnalyzer(
        model_path=args.model_path,
        label_map_path=args.label_map_path,
        device=args.device
    )

    # 执行分析
    results, predictions, aw_masks, tm_masks = analyzer.analyze_ct(args.ct_path, args.output_dir)

    # 打印报告
    analyzer.print_report(results)

    print(f"\n分析完成！结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
