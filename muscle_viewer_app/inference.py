#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肌肉分割推理模块

本模块提供加载训练好的U-Net模型并对CT图像进行肌肉分割的功能。

使用方法:
    from inference import MuscleSegmentor
    segmentor = MuscleSegmentor()
    results = segmentor.segment_ct('/path/to/ct.nii.gz')
"""

import os
import json
import numpy as np
import nibabel as nib
from skimage.transform import resize
import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================================
# 模型定义
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
# 肌肉标签定义
# ============================================================================

MUSCLE_NAMES = {
    'autochthon_left': {'zh': '左侧背部肌群', 'en': 'Left Autochthon'},
    'autochthon_right': {'zh': '右侧背部肌群', 'en': 'Right Autochthon'},
    'gluteus_maximus_left': {'zh': '左臀大肌', 'en': 'Left Gluteus Maximus'},
    'gluteus_maximus_right': {'zh': '右臀大肌', 'en': 'Right Gluteus Maximus'},
    'gluteus_medius_left': {'zh': '左臀中肌', 'en': 'Left Gluteus Medius'},
    'gluteus_medius_right': {'zh': '右臀中肌', 'en': 'Right Gluteus Medius'},
    'gluteus_minimus_left': {'zh': '左臀小肌', 'en': 'Left Gluteus Minimus'},
    'gluteus_minimus_right': {'zh': '右臀小肌', 'en': 'Right Gluteus Minimus'},
    'iliopsoas_left': {'zh': '左髂腰肌', 'en': 'Left Iliopsoas'},
    'iliopsoas_right': {'zh': '右髂腰肌', 'en': 'Right Iliopsoas'},
}

MUSCLE_COLORS = [
    [255, 107, 107],  # 左侧背部肌群 - 红色
    [78, 205, 196],   # 右侧背部肌群 - 青色
    [69, 183, 209],   # 左臀大肌 - 蓝色
    [150, 206, 180],  # 右臀大肌 - 绿色
    [255, 234, 167],  # 左臀中肌 - 黄色
    [221, 160, 221],  # 右臀中肌 - 紫色
    [152, 216, 200],  # 左臀小肌 - 薄荷绿
    [247, 220, 111],  # 右臀小肌 - 金色
    [187, 143, 206],  # 左髂腰肌 - 淡紫色
    [133, 193, 233],  # 右髂腰肌 - 淡蓝色
]


# ============================================================================
# 肌肉分割器
# ============================================================================

class MuscleSegmentor:
    """肌肉分割推理类"""

    def __init__(self, model_path=None, label_map_path=None, device=None):
        """
        初始化分割器

        Args:
            model_path: 模型权重文件路径
            label_map_path: 标签映射文件路径
            device: 计算设备 (cuda/cpu)
        """
        # 获取默认路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, 'models', 'best_model.pth')
        if label_map_path is None:
            label_map_path = os.path.join(base_dir, 'models', 'label_map.json')

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载标签映射
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.num_classes = len(self.label_map)

        # 创建反向映射
        self.idx_to_label = {v: k.replace('.nii.gz', '') for k, v in self.label_map.items()}

        # 加载模型
        self.model = UNet2D(in_ch=1, out_ch=self.num_classes, features=[32, 64, 128, 256])
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 目标图像尺寸
        self.target_shape = (256, 256)

    def preprocess_slice(self, slice_img):
        """预处理单个CT切片"""
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
        """对单个切片进行预测"""
        with torch.no_grad():
            slice_tensor = slice_tensor.to(self.device)
            pred = torch.sigmoid(self.model(slice_tensor))
            pred_bin = (pred > 0.5).float()
        return pred_bin[0].cpu().numpy()

    def segment_ct(self, ct_path, progress_callback=None):
        """
        对整个CT图像进行分割

        Args:
            ct_path: CT图像路径 (.nii.gz)
            progress_callback: 进度回调函数 (current, total)

        Returns:
            dict: 包含分割结果和体积信息的字典
        """
        # 加载CT图像
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata().astype(np.float32)
        header = ct_nii.header

        # 获取体素尺寸
        voxel_dims = header.get_zooms()
        voxel_size_mm = float(voxel_dims[0]) * float(voxel_dims[1])
        slice_thickness = float(voxel_dims[2]) if len(voxel_dims) > 2 else 1.0
        voxel_volume_mm3 = voxel_size_mm * slice_thickness

        # 初始化结果存储
        depth = ct_data.shape[2]
        original_shape = ct_data.shape[:2]
        all_predictions = np.zeros((self.num_classes, *original_shape, depth), dtype=np.float32)

        # 逐层分割
        for z in range(depth):
            slice_img = ct_data[:, :, z]
            slice_tensor = self.preprocess_slice(slice_img)
            pred_mask = self.predict_slice(slice_tensor)

            for c in range(self.num_classes):
                pred_resized = resize(pred_mask[c], original_shape, order=0, preserve_range=True, anti_aliasing=False)
                all_predictions[c, :, :, z] = (pred_resized > 0.5).astype(np.float32)

            if progress_callback:
                progress_callback(z + 1, depth)

        # 计算体积
        volumes = {}
        for c in range(self.num_classes):
            muscle_name = self.idx_to_label[c]
            total_pixels = all_predictions[c].sum()
            volume_mm3 = total_pixels * voxel_volume_mm3
            volume_cm3 = volume_mm3 / 1000

            volumes[muscle_name] = {
                'zh_name': MUSCLE_NAMES[muscle_name]['zh'],
                'en_name': MUSCLE_NAMES[muscle_name]['en'],
                'volume_cm3': float(volume_cm3),
                'volume_mm3': float(volume_mm3),
            }

        total_volume = sum(v['volume_cm3'] for v in volumes.values())

        return {
            'ct_data': ct_data,
            'predictions': all_predictions,
            'volumes': volumes,
            'total_volume_cm3': total_volume,
            'voxel_dims': voxel_dims,
            'num_slices': depth,
        }

    def get_slice_visualization(self, ct_slice, pred_slice, alpha=0.5):
        """
        生成单个切片的可视化图像

        Args:
            ct_slice: CT切片 (H, W)
            pred_slice: 预测掩码 (num_classes, H, W)
            alpha: 叠加透明度

        Returns:
            RGB图像 (H, W, 3)
        """
        # 归一化CT切片
        lo, hi = np.percentile(ct_slice, 1), np.percentile(ct_slice, 99)
        ct_norm = np.clip(ct_slice, lo, hi)
        if hi - lo > 0:
            ct_norm = (ct_norm - lo) / (hi - lo)
        else:
            ct_norm = np.zeros_like(ct_norm)

        # 创建RGB图像（灰度CT）
        rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)

        # 叠加肌肉分割
        for c in range(self.num_classes):
            mask = pred_slice[c] > 0.5
            if mask.sum() > 0:
                color = np.array(MUSCLE_COLORS[c]) / 255.0
                for i in range(3):
                    rgb[:, :, i] = np.where(mask, rgb[:, :, i] * (1 - alpha) + color[i] * alpha, rgb[:, :, i])

        return (rgb * 255).astype(np.uint8)


def calculate_volume_from_ct(ct_path, model_path=None, label_map_path=None):
    """
    便捷函数：计算CT图像中各肌肉的体积

    Args:
        ct_path: CT图像路径
        model_path: 模型路径（可选）
        label_map_path: 标签映射路径（可选）

    Returns:
        dict: 体积信息
    """
    segmentor = MuscleSegmentor(model_path, label_map_path)
    results = segmentor.segment_ct(ct_path)
    return results['volumes'], results['total_volume_cm3']


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python inference.py <ct_path>")
        sys.exit(1)

    ct_path = sys.argv[1]
    print(f"加载CT图像: {ct_path}")

    volumes, total = calculate_volume_from_ct(ct_path)

    print("\n肌肉体积分析结果:")
    print("-" * 50)
    for muscle, data in volumes.items():
        print(f"{data['zh_name']}: {data['volume_cm3']:.2f} cm³")
    print("-" * 50)
    print(f"总肌肉体积: {total:.2f} cm³")
