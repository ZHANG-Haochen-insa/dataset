#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肌肉量分析脚本

本脚本用于对CT图像进行肌肉分割并计算各类肌肉的面积和体积。

功能:
1. 加载训练好的U-Net模型
2. 对CT图像进行逐层分割
3. 计算每种肌肉的面积（每层）和总体积
4. 生成分析报告和可视化结果

使用方法:
    python muscle_analysis.py --ct_path /path/to/ct.nii.gz --output_dir /path/to/output

作者: hzhang02
日期: 2024-12-17
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from skimage.transform import resize
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from datetime import datetime
from tqdm import tqdm


# ============================================================================
# 1. 模型定义（与训练时保持一致）
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

    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 编码器
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器
        rev = list(reversed(features))
        up_in = features[-1] * 2
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
# 2. 肌肉标签定义
# ============================================================================

# 肌肉名称中英法对照
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

# 每种肌肉的颜色（用于可视化）
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
]


# ============================================================================
# 3. 分析类定义
# ============================================================================

class MuscleAnalyzer:
    """肌肉量分析器"""

    def __init__(self, model_path, label_map_path, device=None):
        """
        初始化分析器

        Args:
            model_path: 模型权重文件路径
            label_map_path: 标签映射文件路径
            device: 计算设备 (cuda/cpu)
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        # 加载标签映射
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.num_classes = len(self.label_map)
        print(f"加载标签映射: {self.num_classes} 种肌肉")

        # 创建反向映射
        self.idx_to_label = {v: k.replace('.nii.gz', '') for k, v in self.label_map.items()}

        # 加载模型
        self.model = UNet2D(in_ch=1, out_ch=self.num_classes, features=[32, 64, 128, 256])
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载成功: {model_path}")

        # 目标图像尺寸
        self.target_shape = (256, 256)

    def preprocess_slice(self, slice_img):
        """
        预处理单个CT切片

        Args:
            slice_img: 原始CT切片 (H, W)

        Returns:
            预处理后的tensor (1, 1, H, W)
        """
        # 百分位数窗口化归一化
        lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
        slice_img = np.clip(slice_img, lo, hi)
        if hi - lo > 0:
            slice_img = (slice_img - lo) / (hi - lo)
        else:
            slice_img = np.zeros_like(slice_img)

        # 调整大小
        slice_img = resize(slice_img, self.target_shape, order=1, preserve_range=True, anti_aliasing=True)

        # 转换为tensor
        img_t = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float()
        return img_t

    def predict_slice(self, slice_tensor):
        """
        对单个切片进行预测

        Args:
            slice_tensor: 预处理后的tensor (1, 1, H, W)

        Returns:
            预测掩码 (num_classes, H, W)
        """
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
        print(f"开始分析: {ct_path}")
        print(f"{'=' * 60}")

        # 加载CT图像
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata().astype(np.float32)
        header = ct_nii.header
        affine = ct_nii.affine

        # 获取体素尺寸（用于计算实际体积）
        voxel_dims = header.get_zooms()
        voxel_size_mm = float(voxel_dims[0]) * float(voxel_dims[1])  # 切片内像素面积 (mm²)
        slice_thickness = float(voxel_dims[2]) if len(voxel_dims) > 2 else 1.0  # 切片厚度 (mm)
        voxel_volume_mm3 = voxel_size_mm * slice_thickness  # 体素体积 (mm³)

        print(f"\nCT图像信息:")
        print(f"  形状: {ct_data.shape}")
        print(f"  体素尺寸: {voxel_dims[0]:.2f} x {voxel_dims[1]:.2f} x {slice_thickness:.2f} mm")
        print(f"  切片数量: {ct_data.shape[2]}")

        # 计算缩放因子（从256x256到原始尺寸）
        original_shape = ct_data.shape[:2]
        scale_factor = (original_shape[0] / self.target_shape[0]) * (original_shape[1] / self.target_shape[1])

        # 初始化结果存储
        depth = ct_data.shape[2]
        all_predictions = np.zeros((self.num_classes, *original_shape, depth), dtype=np.float32)
        slice_areas = {i: [] for i in range(self.num_classes)}  # 每层面积

        # 逐层分割
        print(f"\n开始逐层分割...")
        for z in tqdm(range(depth), desc="处理切片"):
            slice_img = ct_data[:, :, z]

            # 预处理和预测
            slice_tensor = self.preprocess_slice(slice_img)
            pred_mask = self.predict_slice(slice_tensor)

            # 将预测结果调整回原始尺寸
            for c in range(self.num_classes):
                pred_resized = resize(pred_mask[c], original_shape, order=0, preserve_range=True, anti_aliasing=False)
                all_predictions[c, :, :, z] = (pred_resized > 0.5).astype(np.float32)

                # 计算该层该肌肉的面积（像素数 * 像素面积）
                pixel_count = all_predictions[c, :, :, z].sum()
                area_mm2 = pixel_count * voxel_size_mm
                slice_areas[c].append(area_mm2)

        # 计算总体积
        print(f"\n计算肌肉体积...")
        results = {
            'ct_path': ct_path,
            'ct_shape': list(ct_data.shape),
            'voxel_dims_mm': [float(v) for v in voxel_dims],
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'muscles': {}
        }

        total_muscle_volume = 0
        for c in range(self.num_classes):
            muscle_name = self.idx_to_label[c]
            muscle_zh = MUSCLE_NAMES[muscle_name]['zh']

            # 计算体积
            total_pixels = all_predictions[c].sum()
            volume_mm3 = total_pixels * voxel_volume_mm3
            volume_cm3 = volume_mm3 / 1000  # 转换为 cm³

            # 计算平均面积
            non_zero_areas = [a for a in slice_areas[c] if a > 0]
            avg_area_mm2 = np.mean(non_zero_areas) if non_zero_areas else 0
            max_area_mm2 = max(slice_areas[c]) if slice_areas[c] else 0

            # 找到肌肉出现的切片范围
            slices_with_muscle = [i for i, a in enumerate(slice_areas[c]) if a > 0]
            slice_range = [min(slices_with_muscle), max(slices_with_muscle)] if slices_with_muscle else [0, 0]

            results['muscles'][muscle_name] = {
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

            total_muscle_volume += volume_cm3

        results['total_muscle_volume_cm3'] = float(total_muscle_volume)

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._save_results(results, all_predictions, ct_data, ct_nii, output_dir)

        return results, all_predictions

    def _save_results(self, results, predictions, ct_data, ct_nii, output_dir):
        """保存分析结果"""

        print(f"\n保存结果到: {output_dir}")

        # 1. 保存JSON报告
        json_path = os.path.join(output_dir, 'muscle_analysis_results.json')
        # 创建一个不包含slice_areas的版本（太大了）
        results_summary = {k: v for k, v in results.items() if k != 'muscles'}
        results_summary['muscles'] = {}
        for muscle_name, muscle_data in results['muscles'].items():
            results_summary['muscles'][muscle_name] = {k: v for k, v in muscle_data.items() if k != 'slice_areas_mm2'}

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"  JSON报告: {json_path}")

        # 2. 保存CSV报告
        csv_path = os.path.join(output_dir, 'muscle_volumes.csv')
        df_data = []
        for muscle_name, muscle_data in results['muscles'].items():
            df_data.append({
                '肌肉名称(中文)': muscle_data['zh_name'],
                '肌肉名称(英文)': muscle_data['en_name'],
                '体积(cm³)': f"{muscle_data['volume_cm3']:.2f}",
                '体积(mm³)': f"{muscle_data['volume_mm3']:.2f}",
                '平均面积(mm²)': f"{muscle_data['avg_area_mm2']:.2f}",
                '最大面积(mm²)': f"{muscle_data['max_area_mm2']:.2f}",
                '出现切片数': muscle_data['num_slices'],
                '切片范围': f"{muscle_data['slice_range'][0]}-{muscle_data['slice_range'][1]}"
            })
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  CSV报告: {csv_path}")

        # 3. 保存分割结果为NIfTI
        # 合并所有通道为单个标签图
        combined_mask = np.zeros(predictions.shape[1:], dtype=np.int16)
        for c in range(self.num_classes):
            combined_mask[predictions[c] > 0.5] = c + 1  # 标签从1开始

        mask_nii = nib.Nifti1Image(combined_mask, ct_nii.affine, ct_nii.header)
        mask_path = os.path.join(output_dir, 'segmentation_mask.nii.gz')
        nib.save(mask_nii, mask_path)
        print(f"  分割掩码: {mask_path}")

        # 4. 生成可视化图像
        self._generate_visualizations(results, predictions, ct_data, output_dir)

    def _generate_visualizations(self, results, predictions, ct_data, output_dir):
        """生成可视化图像"""

        # 图1: 肌肉体积条形图
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        muscles = []
        volumes = []
        colors = []
        for i, (muscle_name, muscle_data) in enumerate(results['muscles'].items()):
            muscles.append(muscle_data['zh_name'])
            volumes.append(muscle_data['volume_cm3'])
            colors.append(MUSCLE_COLORS[i % len(MUSCLE_COLORS)])

        bars = ax1.barh(muscles, volumes, color=colors)
        ax1.set_xlabel('体积 (cm³)', fontsize=12)
        ax1.set_title('各肌肉体积分析', fontsize=14)
        ax1.grid(axis='x', alpha=0.3)

        # 在条形上添加数值
        for bar, vol in zip(bars, volumes):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{vol:.1f}', va='center', fontsize=10)

        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'muscle_volumes_chart.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  体积图表: muscle_volumes_chart.png")

        # 图2: 选择中间切片进行可视化
        depth = ct_data.shape[2]
        sample_slices = [depth // 4, depth // 2, 3 * depth // 4]

        fig2, axes = plt.subplots(2, 3, figsize=(15, 10))

        for idx, z in enumerate(sample_slices):
            # 上排：原始CT
            axes[0, idx].imshow(ct_data[:, :, z].T, cmap='gray', origin='lower')
            axes[0, idx].set_title(f'CT 切片 {z}', fontsize=12)
            axes[0, idx].axis('off')

            # 下排：叠加分割结果
            axes[1, idx].imshow(ct_data[:, :, z].T, cmap='gray', origin='lower')

            # 叠加每种肌肉的轮廓
            for c in range(self.num_classes):
                mask = predictions[c, :, :, z].T
                if mask.sum() > 0:
                    # 使用轮廓显示
                    axes[1, idx].contour(mask, levels=[0.5], colors=[MUSCLE_COLORS[c]], linewidths=1.5)
                    # 半透明填充
                    masked = np.ma.masked_where(mask < 0.5, mask)
                    axes[1, idx].imshow(masked, alpha=0.3, cmap=ListedColormap([MUSCLE_COLORS[c]]), origin='lower')

            axes[1, idx].set_title(f'分割结果 切片 {z}', fontsize=12)
            axes[1, idx].axis('off')

        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'segmentation_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  分割可视化: segmentation_visualization.png")

        # 图3: 肌肉面积沿切片分布
        fig3, ax3 = plt.subplots(figsize=(14, 8))

        for c in range(self.num_classes):
            muscle_name = self.idx_to_label[c]
            muscle_zh = MUSCLE_NAMES[muscle_name]['zh']
            areas = results['muscles'][muscle_name]['slice_areas_mm2']
            ax3.plot(range(len(areas)), areas, label=muscle_zh, color=MUSCLE_COLORS[c], linewidth=1.5)

        ax3.set_xlabel('切片编号', fontsize=12)
        ax3.set_ylabel('面积 (mm²)', fontsize=12)
        ax3.set_title('各肌肉面积沿切片分布', fontsize=14)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'muscle_area_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  面积分布图: muscle_area_distribution.png")

    def print_report(self, results):
        """打印分析报告"""

        print(f"\n{'=' * 60}")
        print("肌肉量分析报告")
        print(f"{'=' * 60}")
        print(f"\n分析时间: {results['analysis_time']}")
        print(f"CT文件: {results['ct_path']}")
        print(f"图像尺寸: {results['ct_shape']}")
        print(f"体素尺寸: {results['voxel_dims_mm'][0]:.2f} x {results['voxel_dims_mm'][1]:.2f} x {results['voxel_dims_mm'][2]:.2f} mm")

        print(f"\n{'=' * 60}")
        print("各肌肉体积统计")
        print(f"{'=' * 60}")
        print(f"\n{'肌肉名称':<20} {'体积(cm³)':>12} {'平均面积(mm²)':>15} {'切片数':>8}")
        print("-" * 60)

        for muscle_name, muscle_data in results['muscles'].items():
            print(f"{muscle_data['zh_name']:<20} {muscle_data['volume_cm3']:>12.2f} "
                  f"{muscle_data['avg_area_mm2']:>15.2f} {muscle_data['num_slices']:>8}")

        print("-" * 60)
        print(f"{'总肌肉体积':<20} {results['total_muscle_volume_cm3']:>12.2f} cm³")
        print(f"{'=' * 60}")

        # 左右对比
        print(f"\n{'=' * 60}")
        print("左右侧肌肉对比")
        print(f"{'=' * 60}")

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
            left_vol = results['muscles'][left]['volume_cm3']
            right_vol = results['muscles'][right]['volume_cm3']
            diff = left_vol - right_vol
            avg = (left_vol + right_vol) / 2
            diff_pct = (diff / avg * 100) if avg > 0 else 0

            print(f"{name:<12} {left_vol:>12.2f} {right_vol:>12.2f} {diff:>+12.2f} {diff_pct:>+11.1f}%")

        print(f"{'=' * 60}")


# ============================================================================
# 4. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CT图像肌肉量分析')
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

    # 检查输入文件是否存在
    if not os.path.exists(args.ct_path):
        print(f"错误: CT文件不存在: {args.ct_path}")
        return

    # 设置输出目录
    if args.output_dir is None:
        ct_dir = os.path.dirname(args.ct_path)
        args.output_dir = os.path.join(ct_dir, 'muscle_analysis')

    # 创建分析器
    analyzer = MuscleAnalyzer(
        model_path=args.model_path,
        label_map_path=args.label_map_path,
        device=args.device
    )

    # 执行分析
    results, predictions = analyzer.analyze_ct(args.ct_path, args.output_dir)

    # 打印报告
    analyzer.print_report(results)

    print(f"\n分析完成！结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
