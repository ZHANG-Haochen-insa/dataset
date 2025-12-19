#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成训练报告所需的可视化图表

包含:
1. 训练过程曲线图（增强版）
2. 训练流程示意图
3. 模型架构示意图
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs'
DOCS_DIR = '/local/hzhang02/data/dataset/docs'
FIGURES_DIR = '/local/hzhang02/data/dataset/docs/figures'

# 创建目录（如果不存在）
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# 1. 加载训练历史数据
# =============================================================================

history_path = os.path.join(OUTPUT_DIR, 'training_history_enhanced.json')
with open(history_path, 'r') as f:
    history = json.load(f)

print("训练历史数据已加载")
print(f"训练轮数: {len(history['train_loss'])}")

# =============================================================================
# 2. 训练过程可视化（增强版 - 中文标签）
# =============================================================================

def create_training_curves_chinese():
    """创建带有中文标签的训练曲线图"""

    actual_epochs = len(history['train_loss'])
    epochs = range(1, actual_epochs + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 颜色配置
    train_color = '#2196F3'  # 蓝色
    val_color = '#FF9800'    # 橙色

    # 1. 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], marker='o', label='Training Loss',
                    linewidth=2.5, color=train_color, markersize=8)
    axes[0, 0].plot(epochs, history['val_loss'], marker='s', label='Validation Loss',
                    linewidth=2.5, color=val_color, markersize=8)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curve / 损失曲线', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 2. Dice系数曲线
    axes[0, 1].plot(epochs, history['train_dice'], marker='o', label='Training Dice',
                    linewidth=2.5, color=train_color, markersize=8)
    axes[0, 1].plot(epochs, history['val_dice'], marker='s', label='Validation Dice',
                    linewidth=2.5, color=val_color, markersize=8)
    axes[0, 1].axhline(y=0.93, color='green', linestyle='--', alpha=0.7, label='Target (0.93)')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Dice Score', fontsize=12)
    axes[0, 1].set_title('Dice Score / Dice系数', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0.5, 1.0])

    # 3. IoU曲线
    axes[0, 2].plot(epochs, history['val_iou'], marker='d', linewidth=2.5,
                    color='#9C27B0', markersize=8)
    axes[0, 2].fill_between(epochs, history['val_iou'], alpha=0.3, color='#9C27B0')
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('IoU Score', fontsize=12)
    axes[0, 2].set_title('Validation IoU / 验证IoU', fontsize=14, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].set_ylim([0.5, 1.0])

    # 4. 过拟合分析
    overfit_gaps = [t - v for t, v in zip(history['train_dice'], history['val_dice'])]
    colors = ['#4CAF50' if g < 0.05 else '#FF9800' if g < 0.1 else '#F44336' for g in overfit_gaps]
    axes[1, 0].bar(epochs, overfit_gaps, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 0].axhline(y=0.05, color='#4CAF50', linestyle='--', alpha=0.7, label='Good (<0.05)')
    axes[1, 0].axhline(y=0.1, color='#F44336', linestyle='--', alpha=0.7, label='Overfit (>0.1)')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Train Dice - Val Dice', fontsize=12)
    axes[1, 0].set_title('Overfitting Analysis / 过拟合分析', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3, axis='y')

    # 5. 梯度范数
    axes[1, 1].plot(epochs, history['gradient_norm'], marker='o', linewidth=2.5,
                    color='#795548', markersize=8)
    axes[1, 1].fill_between(epochs, history['gradient_norm'], alpha=0.3, color='#795548')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Gradient Norm', fontsize=12)
    axes[1, 1].set_title('Gradient Norm / 梯度范数', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # 6. 学习率
    axes[1, 2].plot(epochs, history['learning_rate'], marker='o', linewidth=2.5,
                    color='#00BCD4', markersize=8)
    axes[1, 2].fill_between(epochs, history['learning_rate'], alpha=0.3, color='#00BCD4')
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 2].set_title('Learning Rate Schedule / 学习率调度', fontsize=14, fontweight='bold')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()

    # 保存到figures目录
    save_path = os.path.join(FIGURES_DIR, 'training_curves_chinese.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"训练曲线图已保存: {save_path}")

    return save_path

# =============================================================================
# 3. 训练流程示意图
# =============================================================================

def create_training_pipeline_diagram():
    """创建训练流程示意图"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 定义颜色
    colors = {
        'data': '#E3F2FD',      # 浅蓝
        'process': '#FFF3E0',   # 浅橙
        'model': '#E8F5E9',     # 浅绿
        'output': '#FCE4EC',    # 浅粉
        'arrow': '#424242'      # 深灰
    }

    # 定义框的样式
    box_style = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', linewidth=2)

    def draw_box(x, y, width, height, text, color, fontsize=10):
        """绘制圆角矩形框"""
        box = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='#333333', linewidth=2)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_arrow(start, end, color='#424242'):
        """绘制箭头"""
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # 标题
    ax.text(8, 9.5, 'Training Pipeline / 训练流程图', ha='center', va='center',
            fontsize=18, fontweight='bold')

    # 第一行: 数据准备
    draw_box(0.5, 7, 2.5, 1.2, 'CT Images\nCT图像\n(.nii.gz)', colors['data'])
    draw_box(3.5, 7, 2.5, 1.2, 'Segmentation\nMasks\n分割标签', colors['data'])
    draw_box(6.5, 7, 2.5, 1.2, 'Data Loading\n数据加载\nSliceDataset', colors['process'])
    draw_box(9.5, 7, 2.5, 1.2, 'Preprocessing\n预处理\n归一化/缩放', colors['process'])
    draw_box(12.5, 7, 3, 1.2, 'DataLoader\n批量加载\n(batch=16)', colors['process'])

    # 箭头连接第一行
    draw_arrow((3, 7.6), (3.5, 7.6))
    draw_arrow((6, 7.6), (6.5, 7.6))
    draw_arrow((9, 7.6), (9.5, 7.6))
    draw_arrow((12, 7.6), (12.5, 7.6))

    # 第二行: 模型训练
    draw_box(1, 4.5, 3, 1.5, 'U-Net 2D\n编码器-解码器\n[32,64,128,256]', colors['model'])
    draw_box(4.5, 4.5, 2.5, 1.5, 'Forward Pass\n前向传播\n预测输出', colors['process'])
    draw_box(7.5, 4.5, 2.5, 1.5, 'Loss Function\n损失函数\nBCEWithLogits', colors['process'])
    draw_box(10.5, 4.5, 2.5, 1.5, 'Backward Pass\n反向传播\n梯度计算', colors['process'])
    draw_box(13.5, 4.5, 2, 1.5, 'Optimizer\n优化器\nAdam', colors['process'])

    # 箭头连接第二行
    draw_arrow((4, 5.25), (4.5, 5.25))
    draw_arrow((7, 5.25), (7.5, 5.25))
    draw_arrow((10, 5.25), (10.5, 5.25))
    draw_arrow((13, 5.25), (13.5, 5.25))

    # 垂直箭头
    draw_arrow((14, 7), (14, 6))
    draw_arrow((2.5, 6), (2.5, 4.5))

    # 第三行: 评估和输出
    draw_box(1, 2, 2.5, 1.2, 'Validation\n验证\n评估性能', colors['process'])
    draw_box(4, 2, 2.5, 1.2, 'Metrics\n评估指标\nDice/IoU', colors['output'])
    draw_box(7, 2, 2.5, 1.2, 'Early Stopping\n早停机制\nPatience=5', colors['process'])
    draw_box(10, 2, 2.5, 1.2, 'LR Schedule\n学习率调度\nCosine', colors['process'])
    draw_box(13, 2, 2.5, 1.2, 'Checkpoints\n检查点保存\n.pth文件', colors['output'])

    # 箭头连接第三行
    draw_arrow((3.5, 2.6), (4, 2.6))
    draw_arrow((6.5, 2.6), (7, 2.6))
    draw_arrow((9.5, 2.6), (10, 2.6))
    draw_arrow((12.5, 2.6), (13, 2.6))

    # 循环箭头（训练循环）
    ax.annotate('', xy=(1, 5.25), xytext=(1, 2.6),
               arrowprops=dict(arrowstyle='->', color='#F44336', lw=2,
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(0.3, 4, 'Epoch\nLoop', fontsize=9, color='#F44336', fontweight='bold')

    # 最终输出
    draw_box(6, 0.3, 4, 1, 'Final Model / 最终模型\nbest_model.pth', colors['output'])
    draw_arrow((8, 2), (8, 1.3))

    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'training_pipeline.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"训练流程图已保存: {save_path}")

    return save_path

# =============================================================================
# 4. U-Net架构示意图
# =============================================================================

def create_unet_architecture_diagram():
    """创建U-Net架构示意图"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 颜色
    encoder_color = '#BBDEFB'  # 浅蓝
    decoder_color = '#C8E6C9'  # 浅绿
    bottleneck_color = '#FFE0B2'  # 浅橙
    skip_color = '#E1BEE7'     # 浅紫

    def draw_block(x, y, w, h, text, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # 标题
    ax.text(7, 9.5, 'U-Net Architecture / U-Net架构', ha='center', fontsize=16, fontweight='bold')

    # 编码器 (左侧，从上到下)
    # 输入
    draw_block(0.5, 7.5, 1.5, 1, 'Input\n1×256×256', '#E0E0E0')

    # 编码层
    draw_block(0.5, 6, 1.5, 1, 'Conv 32\n256×256', encoder_color)
    draw_block(0.5, 4.5, 1.5, 1, 'Conv 64\n128×128', encoder_color)
    draw_block(0.5, 3, 1.5, 1, 'Conv 128\n64×64', encoder_color)
    draw_block(0.5, 1.5, 1.5, 1, 'Conv 256\n32×32', encoder_color)

    # 池化箭头
    for y in [7.5, 6, 4.5, 3]:
        ax.annotate('', xy=(1.25, y), xytext=(1.25, y+0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(1.7, y+0.2, 'Pool', fontsize=7)

    # 瓶颈层
    draw_block(3, 0.3, 2, 1, 'Bottleneck\nConv 512\n16×16', bottleneck_color)
    ax.annotate('', xy=(3, 0.8), xytext=(2, 1.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 解码器 (右侧，从下到上)
    draw_block(6, 1.5, 1.5, 1, 'DeConv 256\n32×32', decoder_color)
    draw_block(6, 3, 1.5, 1, 'DeConv 128\n64×64', decoder_color)
    draw_block(6, 4.5, 1.5, 1, 'DeConv 64\n128×128', decoder_color)
    draw_block(6, 6, 1.5, 1, 'DeConv 32\n256×256', decoder_color)

    # 上采样箭头
    ax.annotate('', xy=(6, 0.8), xytext=(5, 0.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    for y in [1.5, 3, 4.5]:
        ax.annotate('', xy=(6.75, y+1), xytext=(6.75, y+0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(7, y+0.7, 'Up', fontsize=7)

    # Skip Connections
    skip_ys = [(6, 6), (4.5, 4.5), (3, 3), (1.5, 1.5)]
    for (y1, y2) in skip_ys:
        ax.annotate('', xy=(6, y2+0.5), xytext=(2, y1+0.5),
                   arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=2,
                                  connectionstyle='arc3,rad=-0.2', linestyle='dashed'))

    ax.text(4, 7.2, 'Skip Connections', fontsize=10, color='#7B1FA2', fontweight='bold', ha='center')

    # 输出层
    draw_block(6, 7.5, 1.5, 1, 'Conv 1×1\n10×256×256', '#FFCDD2')
    ax.annotate('', xy=(6.75, 7.5), xytext=(6.75, 7),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 最终输出
    draw_block(6, 8.7, 1.5, 0.7, 'Output\n10 classes', '#F8BBD9')
    ax.annotate('', xy=(6.75, 8.7), xytext=(6.75, 8.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=encoder_color, edgecolor='black', label='Encoder / 编码器'),
        plt.Rectangle((0, 0), 1, 1, facecolor=decoder_color, edgecolor='black', label='Decoder / 解码器'),
        plt.Rectangle((0, 0), 1, 1, facecolor=bottleneck_color, edgecolor='black', label='Bottleneck / 瓶颈层'),
        plt.Line2D([0], [0], color='#7B1FA2', linestyle='dashed', lw=2, label='Skip Connection / 跳跃连接'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 参数信息
    info_text = """
    Model Parameters / 模型参数:
    - Total: 7,796,650
    - Input: 1 channel (CT)
    - Output: 10 channels (muscles)
    - Features: [32, 64, 128, 256, 512]
    """
    ax.text(10, 3, info_text, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'unet_architecture.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"U-Net架构图已保存: {save_path}")

    return save_path

# =============================================================================
# 5. 肌肉结构示意图
# =============================================================================

def create_muscle_structure_chart():
    """创建肌肉结构分布图"""

    muscles = {
        'Autochthon L\n左背部肌群': 0,
        'Autochthon R\n右背部肌群': 1,
        'Glut. Max. L\n左臀大肌': 2,
        'Glut. Max. R\n右臀大肌': 3,
        'Glut. Med. L\n左臀中肌': 4,
        'Glut. Med. R\n右臀中肌': 5,
        'Glut. Min. L\n左臀小肌': 6,
        'Glut. Min. R\n右臀小肌': 7,
        'Iliopsoas L\n左髂腰肌': 8,
        'Iliopsoas R\n右髂腰肌': 9,
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # 创建颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 绘制条形图
    bars = ax.barh(list(muscles.keys()), list(muscles.values()), color=colors)

    # 添加通道索引标签
    for bar, (name, idx) in zip(bars, muscles.items()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'Channel {idx}', va='center', fontsize=10)

    ax.set_xlabel('Channel Index / 通道索引', fontsize=12)
    ax.set_title('Target Muscle Structures / 目标肌肉结构 (10 Classes)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 12)

    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'muscle_structures.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"肌肉结构图已保存: {save_path}")

    return save_path

# =============================================================================
# 6. 训练配置信息图
# =============================================================================

def create_training_config_summary():
    """创建训练配置摘要图"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # 1. 训练指标表格
    metrics_data = [
        ['Final Train Dice', f'{history["train_dice"][-1]:.4f}'],
        ['Final Val Dice', f'{history["val_dice"][-1]:.4f}'],
        ['Final Val IoU', f'{history["val_iou"][-1]:.4f}'],
        ['Best Val Dice', f'{max(history["val_dice"]):.4f}'],
        ['Final Train Loss', f'{history["train_loss"][-1]:.4f}'],
        ['Final Val Loss', f'{history["val_loss"][-1]:.4f}'],
    ]

    axes[0].axis('off')
    table1 = axes[0].table(cellText=metrics_data,
                           colLabels=['Metric / 指标', 'Value / 值'],
                           loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.2, 1.8)
    axes[0].set_title('Training Results / 训练结果', fontsize=14, fontweight='bold', pad=20)

    # 2. 训练配置表格
    config_data = [
        ['Epochs', f'{len(history["train_loss"])}'],
        ['Batch Size', '16'],
        ['Learning Rate', '1e-3 → 9e-4'],
        ['Optimizer', 'Adam'],
        ['Scheduler', 'CosineAnnealing'],
        ['Loss Function', 'BCEWithLogits'],
    ]

    axes[1].axis('off')
    table2 = axes[1].table(cellText=config_data,
                           colLabels=['Config / 配置', 'Value / 值'],
                           loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.2, 1.8)
    axes[1].set_title('Training Config / 训练配置', fontsize=14, fontweight='bold', pad=20)

    # 3. 数据集信息表格
    data_info = [
        ['Subjects', '101 (s0000-s0100)'],
        ['Train/Val Split', '80% / 20%'],
        ['Input Size', '256 × 256'],
        ['Output Classes', '10 muscles'],
        ['Train Slices', '~18,000'],
        ['Val Slices', '~4,500'],
    ]

    axes[2].axis('off')
    table3 = axes[2].table(cellText=data_info,
                           colLabels=['Dataset / 数据集', 'Value / 值'],
                           loc='center', cellLoc='center')
    table3.auto_set_font_size(False)
    table3.set_fontsize(11)
    table3.scale(1.2, 1.8)
    axes[2].set_title('Dataset Info / 数据集信息', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'training_config_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"训练配置摘要图已保存: {save_path}")

    return save_path

# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("开始生成训练报告图表")
    print("=" * 60)

    # 生成所有图表
    curves_path = create_training_curves_chinese()
    pipeline_path = create_training_pipeline_diagram()
    arch_path = create_unet_architecture_diagram()
    muscle_path = create_muscle_structure_chart()
    config_path = create_training_config_summary()

    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print("保存位置: " + FIGURES_DIR)
    print("=" * 60)
