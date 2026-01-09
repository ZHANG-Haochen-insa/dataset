#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肌肉内脂肪浸润分析程序 (Intramuscular Adipose Tissue - IMAT Analysis)

功能：
1. 加载训练好的肌肉分割模型和脂肪分割模型
2. 对CT图像进行推理，分别预测肌肉区域和脂肪区域
3. 计算肌肉内脂肪（IMAT）= 脂肪区域 ∩ 肌肉区域
4. 可视化：深色显示肌肉，浅色显示肌肉内脂肪
5. 计算并显示面积比例

作者: hzhang02
日期: 2025-01-08
"""

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

import torch
import torch.nn as nn

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 模型定义 (与训练时保持一致)
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
# 2. 分析器类
# ============================================================================

class MuscleFatAnalyzer:
    """肌肉内脂肪分析器"""

    def __init__(self,
                 muscle_model_path: str,
                 fat_model_path: str,
                 device: str = None):
        """
        初始化分析器

        Args:
            muscle_model_path: 肌肉分割模型路径
            fat_model_path: 脂肪分割模型路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 加载肌肉模型 (输出1通道 - 环腹部肌肉)
        print(f"加载肌肉模型: {muscle_model_path}")
        self.muscle_model = UNet2D(in_ch=1, out_ch=1, features=[32, 64, 128, 256])
        muscle_checkpoint = torch.load(muscle_model_path, map_location=self.device)
        self.muscle_model.load_state_dict(muscle_checkpoint['model_state_dict'])
        self.muscle_model.to(self.device)
        self.muscle_model.eval()
        print(f"  肌肉模型加载成功 (验证Dice: {muscle_checkpoint.get('val_dice', 'N/A'):.4f})")

        # 加载脂肪模型 (输出2通道 - SAT和VAT)
        print(f"加载脂肪模型: {fat_model_path}")
        self.fat_model = UNet2D(in_ch=1, out_ch=2, features=[32, 64, 128, 256])
        fat_checkpoint = torch.load(fat_model_path, map_location=self.device)
        self.fat_model.load_state_dict(fat_checkpoint['model_state_dict'])
        self.fat_model.to(self.device)
        self.fat_model.eval()
        print(f"  脂肪模型加载成功 (验证Dice: {fat_checkpoint.get('val_dice', 'N/A'):.4f})")

        self.target_shape = (256, 256)

    def preprocess_slice(self, slice_img: np.ndarray) -> torch.Tensor:
        """
        预处理CT切片

        Args:
            slice_img: CT切片 (H, W)，原始HU值

        Returns:
            预处理后的tensor (1, 1, H, W)
        """
        # 归一化
        lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
        slice_normalized = np.clip(slice_img, lo, hi)
        if hi - lo > 0:
            slice_normalized = (slice_normalized - lo) / (hi - lo)
        else:
            slice_normalized = np.zeros_like(slice_normalized)

        # 调整大小
        slice_resized = resize(slice_normalized, self.target_shape, order=1,
                               preserve_range=True, anti_aliasing=True)

        # 转换为tensor
        img_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float()
        return img_tensor.to(self.device)

    def predict(self, slice_img: np.ndarray, threshold: float = 0.5):
        """
        对单个CT切片进行预测

        Args:
            slice_img: CT切片 (H, W)，原始HU值
            threshold: 二值化阈值

        Returns:
            dict: 包含各种掩码和统计信息
        """
        original_shape = slice_img.shape
        img_tensor = self.preprocess_slice(slice_img)

        with torch.no_grad():
            # 预测肌肉
            muscle_pred = torch.sigmoid(self.muscle_model(img_tensor))
            muscle_mask = (muscle_pred > threshold).float()[0, 0].cpu().numpy()

            # 预测脂肪 (SAT和VAT)
            fat_pred = torch.sigmoid(self.fat_model(img_tensor))
            sat_mask = (fat_pred[0, 0] > threshold).float().cpu().numpy()
            vat_mask = (fat_pred[0, 1] > threshold).float().cpu().numpy()

            # 合并所有脂肪
            all_fat_mask = np.logical_or(sat_mask, vat_mask).astype(np.float32)

        # 调整回原始大小
        muscle_mask = resize(muscle_mask, original_shape, order=0,
                            preserve_range=True, anti_aliasing=False)
        sat_mask = resize(sat_mask, original_shape, order=0,
                         preserve_range=True, anti_aliasing=False)
        vat_mask = resize(vat_mask, original_shape, order=0,
                         preserve_range=True, anti_aliasing=False)
        all_fat_mask = resize(all_fat_mask, original_shape, order=0,
                             preserve_range=True, anti_aliasing=False)

        # 二值化
        muscle_mask = (muscle_mask > 0.5).astype(np.float32)
        sat_mask = (sat_mask > 0.5).astype(np.float32)
        vat_mask = (vat_mask > 0.5).astype(np.float32)
        all_fat_mask = (all_fat_mask > 0.5).astype(np.float32)

        # 计算肌肉内脂肪 (IMAT) = 脂肪 ∩ 肌肉
        imat_mask = np.logical_and(all_fat_mask, muscle_mask).astype(np.float32)

        # 纯肌肉区域 (排除脂肪浸润的部分)
        pure_muscle_mask = np.logical_and(muscle_mask, ~imat_mask.astype(bool)).astype(np.float32)

        # 计算面积 (像素数)
        muscle_area = muscle_mask.sum()
        imat_area = imat_mask.sum()
        pure_muscle_area = pure_muscle_mask.sum()
        sat_area = sat_mask.sum()
        vat_area = vat_mask.sum()

        # 计算比例
        imat_ratio = (imat_area / muscle_area * 100) if muscle_area > 0 else 0
        pure_muscle_ratio = (pure_muscle_area / muscle_area * 100) if muscle_area > 0 else 0

        return {
            'muscle_mask': muscle_mask,           # 总肌肉区域
            'pure_muscle_mask': pure_muscle_mask, # 纯肌肉（无脂肪浸润）
            'imat_mask': imat_mask,               # 肌肉内脂肪 (IMAT)
            'sat_mask': sat_mask,                 # 皮下脂肪
            'vat_mask': vat_mask,                 # 内脏脂肪
            'all_fat_mask': all_fat_mask,         # 所有脂肪
            'stats': {
                'muscle_area_pixels': int(muscle_area),
                'imat_area_pixels': int(imat_area),
                'pure_muscle_area_pixels': int(pure_muscle_area),
                'sat_area_pixels': int(sat_area),
                'vat_area_pixels': int(vat_area),
                'imat_ratio_percent': float(imat_ratio),
                'pure_muscle_ratio_percent': float(pure_muscle_ratio),
            }
        }

    def visualize(self, slice_img: np.ndarray, result: dict,
                  save_path: str = None, show: bool = True,
                  title: str = None):
        """
        可视化分析结果

        Args:
            slice_img: 原始CT切片
            result: predict()返回的结果字典
            save_path: 保存路径
            show: 是否显示图像
            title: 图像标题
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 归一化CT图像用于显示
        lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
        ct_display = np.clip(slice_img, lo, hi)
        ct_display = (ct_display - lo) / (hi - lo + 1e-8)

        stats = result['stats']

        # ===== 第一行 =====

        # 1. 原始CT图像
        axes[0, 0].imshow(ct_display, cmap='gray')
        axes[0, 0].set_title('CT Image', fontsize=14)
        axes[0, 0].axis('off')

        # 2. 肌肉区域（深色）+ 肌肉内脂肪（浅色/黄色）
        axes[0, 1].imshow(ct_display, cmap='gray')

        # 创建叠加掩码：深红色为纯肌肉，黄色为IMAT
        muscle_overlay = np.zeros((*slice_img.shape, 4))  # RGBA

        # 纯肌肉 - 深红色 (暗色)
        pure_muscle = result['pure_muscle_mask']
        muscle_overlay[pure_muscle > 0.5] = [0.6, 0.1, 0.1, 0.6]  # 深红色

        # IMAT - 黄色 (浅色/亮色)
        imat = result['imat_mask']
        muscle_overlay[imat > 0.5] = [1.0, 0.9, 0.2, 0.7]  # 亮黄色

        axes[0, 1].imshow(muscle_overlay)
        axes[0, 1].set_title(f'Muscle (Dark) + IMAT (Light)\n'
                            f'IMAT Ratio: {stats["imat_ratio_percent"]:.1f}%',
                            fontsize=14)
        axes[0, 1].axis('off')

        # 添加图例
        dark_patch = mpatches.Patch(color=(0.6, 0.1, 0.1, 0.6), label='Pure Muscle')
        light_patch = mpatches.Patch(color=(1.0, 0.9, 0.2, 0.7), label='IMAT (Fat in Muscle)')
        axes[0, 1].legend(handles=[dark_patch, light_patch], loc='upper right', fontsize=10)

        # 3. 所有分割结果叠加
        axes[0, 2].imshow(ct_display, cmap='gray')

        # 多区域叠加
        combined_overlay = np.zeros((*slice_img.shape, 4))

        # SAT - 蓝色
        combined_overlay[result['sat_mask'] > 0.5] = [0.2, 0.4, 0.8, 0.5]

        # VAT - 青色
        combined_overlay[result['vat_mask'] > 0.5] = [0.2, 0.8, 0.8, 0.5]

        # 纯肌肉 - 红色
        combined_overlay[pure_muscle > 0.5] = [0.8, 0.2, 0.2, 0.5]

        # IMAT - 黄色 (覆盖在最上面)
        combined_overlay[imat > 0.5] = [1.0, 0.9, 0.2, 0.7]

        axes[0, 2].imshow(combined_overlay)
        axes[0, 2].set_title('All Segmentations', fontsize=14)
        axes[0, 2].axis('off')

        # 图例
        sat_patch = mpatches.Patch(color=(0.2, 0.4, 0.8, 0.5), label='SAT')
        vat_patch = mpatches.Patch(color=(0.2, 0.8, 0.8, 0.5), label='VAT')
        muscle_patch = mpatches.Patch(color=(0.8, 0.2, 0.2, 0.5), label='Muscle')
        imat_patch = mpatches.Patch(color=(1.0, 0.9, 0.2, 0.7), label='IMAT')
        axes[0, 2].legend(handles=[sat_patch, vat_patch, muscle_patch, imat_patch],
                         loc='upper right', fontsize=9)

        # ===== 第二行 =====

        # 4. 肌肉掩码
        axes[1, 0].imshow(ct_display, cmap='gray')
        axes[1, 0].imshow(result['muscle_mask'], alpha=0.5, cmap='Reds')
        axes[1, 0].set_title(f'Muscle Region\n'
                            f'Area: {stats["muscle_area_pixels"]:,} px',
                            fontsize=14)
        axes[1, 0].axis('off')

        # 5. IMAT详细视图
        axes[1, 1].imshow(ct_display, cmap='gray')
        axes[1, 1].imshow(result['imat_mask'], alpha=0.7, cmap='YlOrRd')
        axes[1, 1].set_title(f'IMAT (Intramuscular Fat)\n'
                            f'Area: {stats["imat_area_pixels"]:,} px',
                            fontsize=14)
        axes[1, 1].axis('off')

        # 6. 面积统计饼图
        areas = [
            stats['pure_muscle_area_pixels'],
            stats['imat_area_pixels'],
        ]
        labels = [
            f'Pure Muscle\n({stats["pure_muscle_ratio_percent"]:.1f}%)',
            f'IMAT\n({stats["imat_ratio_percent"]:.1f}%)',
        ]
        colors = ['#8B0000', '#FFD700']  # 深红和金黄

        # 只有当有数据时才绘制饼图
        if sum(areas) > 0:
            wedges, texts, autotexts = axes[1, 2].pie(
                areas, labels=labels, colors=colors,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                startangle=90,
                explode=(0, 0.05),
                shadow=True
            )
            axes[1, 2].set_title('Muscle Composition\n(Area Distribution)', fontsize=14)
        else:
            axes[1, 2].text(0.5, 0.5, 'No muscle detected',
                           ha='center', va='center', fontsize=14)
            axes[1, 2].set_title('Muscle Composition', fontsize=14)

        # 添加总标题
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"图像已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def analyze_volume(self, ct_path: str, slice_indices: list = None,
                       output_dir: str = None):
        """
        分析整个CT体积

        Args:
            ct_path: CT文件路径 (.nii.gz)
            slice_indices: 要分析的切片索引列表，None表示均匀采样
            output_dir: 输出目录

        Returns:
            dict: 包含所有切片的统计信息
        """
        print(f"\n加载CT数据: {ct_path}")
        img = nib.load(ct_path)
        ct_data = img.get_fdata().astype(np.float32)
        print(f"CT形状: {ct_data.shape}")

        depth = ct_data.shape[2]

        if slice_indices is None:
            # 均匀采样10个切片
            slice_indices = np.linspace(depth * 0.2, depth * 0.8, 10, dtype=int)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        all_stats = []

        for i, z in enumerate(slice_indices):
            if z < 0 or z >= depth:
                continue

            print(f"\n处理切片 {z}/{depth} ({i+1}/{len(slice_indices)})...")

            slice_img = ct_data[:, :, z]
            result = self.predict(slice_img)

            stats = result['stats']
            stats['slice_index'] = int(z)
            all_stats.append(stats)

            print(f"  肌肉面积: {stats['muscle_area_pixels']:,} px")
            print(f"  IMAT面积: {stats['imat_area_pixels']:,} px")
            print(f"  IMAT比例: {stats['imat_ratio_percent']:.2f}%")

            if output_dir:
                save_path = os.path.join(output_dir, f'slice_{z:04d}_analysis.png')
                self.visualize(slice_img, result, save_path=save_path, show=False,
                              title=f'Slice {z} Analysis')

        # 计算总体统计
        total_muscle = sum(s['muscle_area_pixels'] for s in all_stats)
        total_imat = sum(s['imat_area_pixels'] for s in all_stats)
        avg_imat_ratio = np.mean([s['imat_ratio_percent'] for s in all_stats])

        summary = {
            'slices_analyzed': len(all_stats),
            'total_muscle_pixels': total_muscle,
            'total_imat_pixels': total_imat,
            'average_imat_ratio_percent': float(avg_imat_ratio),
            'per_slice_stats': all_stats,
        }

        if output_dir:
            # 保存统计信息
            stats_path = os.path.join(output_dir, 'analysis_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n统计信息已保存: {stats_path}")

            # 绘制IMAT比例变化图
            self._plot_imat_trend(all_stats, output_dir)

        return summary

    def _plot_imat_trend(self, all_stats: list, output_dir: str):
        """绘制IMAT比例随切片变化的趋势图"""
        slices = [s['slice_index'] for s in all_stats]
        imat_ratios = [s['imat_ratio_percent'] for s in all_stats]
        muscle_areas = [s['muscle_area_pixels'] for s in all_stats]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # IMAT比例趋势
        axes[0].plot(slices, imat_ratios, 'o-', linewidth=2, markersize=8, color='gold')
        axes[0].fill_between(slices, imat_ratios, alpha=0.3, color='gold')
        axes[0].set_xlabel('Slice Index', fontsize=12)
        axes[0].set_ylabel('IMAT Ratio (%)', fontsize=12)
        axes[0].set_title('IMAT Ratio by Slice', fontsize=14)
        axes[0].grid(alpha=0.3)
        axes[0].axhline(y=np.mean(imat_ratios), color='red', linestyle='--',
                       label=f'Mean: {np.mean(imat_ratios):.1f}%')
        axes[0].legend()

        # 肌肉面积趋势
        axes[1].plot(slices, muscle_areas, 's-', linewidth=2, markersize=8, color='darkred')
        axes[1].fill_between(slices, muscle_areas, alpha=0.3, color='darkred')
        axes[1].set_xlabel('Slice Index', fontsize=12)
        axes[1].set_ylabel('Muscle Area (pixels)', fontsize=12)
        axes[1].set_title('Muscle Area by Slice', fontsize=14)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'imat_trend.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"趋势图已保存: {save_path}")


# ============================================================================
# 3. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='肌肉内脂肪浸润分析')
    parser.add_argument('--ct', type=str, help='CT文件路径 (.nii.gz)')
    parser.add_argument('--slice', type=int, default=None, help='指定分析的切片索引')
    parser.add_argument('--output', type=str, default='./analysis_output', help='输出目录')
    parser.add_argument('--show', action='store_true', help='显示图像')
    parser.add_argument('--demo', action='store_true', help='运行演示模式')

    args = parser.parse_args()

    # 模型路径
    BASE_DIR = '/local/hzhang02/data/dataset'
    MUSCLE_MODEL_PATH = os.path.join(BASE_DIR, 'outputs_abdominal_wall', 'best_abdominal_wall_model.pth')
    FAT_MODEL_PATH = os.path.join(BASE_DIR, 'outputs_fat', 'best_fat_model.pth')

    # 检查模型文件
    if not os.path.exists(MUSCLE_MODEL_PATH):
        print(f"错误: 肌肉模型文件不存在: {MUSCLE_MODEL_PATH}")
        return
    if not os.path.exists(FAT_MODEL_PATH):
        print(f"错误: 脂肪模型文件不存在: {FAT_MODEL_PATH}")
        return

    # 初始化分析器
    analyzer = MuscleFatAnalyzer(MUSCLE_MODEL_PATH, FAT_MODEL_PATH)

    if args.demo or args.ct is None:
        # 演示模式：使用第一个可用的受试者
        print("\n运行演示模式...")
        data_root = '/local/hzhang02/data'
        demo_ct = None

        for i in range(50):
            ct_path = os.path.join(data_root, f's{i:04d}', 'ct.nii.gz')
            if os.path.exists(ct_path):
                demo_ct = ct_path
                break

        if demo_ct is None:
            print("错误: 未找到可用的CT数据")
            return

        print(f"使用演示数据: {demo_ct}")

        # 加载CT
        img = nib.load(demo_ct)
        ct_data = img.get_fdata().astype(np.float32)
        depth = ct_data.shape[2]

        # 分析中间切片
        slice_idx = depth // 2
        print(f"\n分析切片 {slice_idx}...")

        slice_img = ct_data[:, :, slice_idx]
        result = analyzer.predict(slice_img)

        # 打印统计信息
        stats = result['stats']
        print("\n" + "=" * 50)
        print("分析结果:")
        print("=" * 50)
        print(f"  肌肉总面积: {stats['muscle_area_pixels']:,} 像素")
        print(f"  纯肌肉面积: {stats['pure_muscle_area_pixels']:,} 像素 ({stats['pure_muscle_ratio_percent']:.1f}%)")
        print(f"  IMAT面积: {stats['imat_area_pixels']:,} 像素 ({stats['imat_ratio_percent']:.1f}%)")
        print(f"  皮下脂肪面积: {stats['sat_area_pixels']:,} 像素")
        print(f"  内脏脂肪面积: {stats['vat_area_pixels']:,} 像素")
        print("=" * 50)

        # 创建输出目录
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

        # 可视化
        save_path = os.path.join(output_dir, 'demo_analysis.png')
        analyzer.visualize(slice_img, result, save_path=save_path,
                          show=args.show, title=f'Demo Analysis - Slice {slice_idx}')

        # 分析多个切片
        print("\n\n分析多个切片...")
        analyzer.analyze_volume(demo_ct, output_dir=output_dir)

    else:
        # 正常模式
        if not os.path.exists(args.ct):
            print(f"错误: CT文件不存在: {args.ct}")
            return

        if args.slice is not None:
            # 分析单个切片
            img = nib.load(args.ct)
            ct_data = img.get_fdata().astype(np.float32)

            slice_img = ct_data[:, :, args.slice]
            result = analyzer.predict(slice_img)

            os.makedirs(args.output, exist_ok=True)
            save_path = os.path.join(args.output, f'slice_{args.slice}_analysis.png')
            analyzer.visualize(slice_img, result, save_path=save_path,
                              show=args.show, title=f'Slice {args.slice} Analysis')
        else:
            # 分析整个体积
            analyzer.analyze_volume(args.ct, output_dir=args.output)

    print("\n分析完成!")


if __name__ == '__main__':
    main()
