#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肌肉分割可视化应用

基于Gradio的Web应用，用于查看CT图像的肌肉分割结果和体积分析。

功能:
1. 加载CT图像 (.nii.gz)
2. 显示CT切片和肌肉分割叠加
3. 计算并显示各肌肉体积
4. 支持切片浏览

运行方法:
    python app.py
"""

import os
import numpy as np
import gradio as gr
from inference import MuscleSegmentor, MUSCLE_NAMES, MUSCLE_COLORS

# 全局变量存储当前分析结果
current_results = None
segmentor = None


def initialize_model():
    """初始化模型"""
    global segmentor
    if segmentor is None:
        segmentor = MuscleSegmentor()
    return segmentor


def analyze_ct(ct_file, progress=gr.Progress()):
    """分析上传的CT文件"""
    global current_results

    if ct_file is None:
        return None, "请上传CT文件", 0, gr.update(maximum=0)

    try:
        # 初始化模型
        seg = initialize_model()

        # 执行分割
        progress(0, desc="正在加载CT图像...")

        def progress_callback(current, total):
            progress(current / total, desc=f"分割中: {current}/{total}")

        results = seg.segment_ct(ct_file.name, progress_callback)
        current_results = results

        # 生成体积报告
        volume_text = generate_volume_report(results)

        # 获取默认切片（中间切片）
        default_slice = results['num_slices'] // 2

        return (
            get_slice_image(default_slice),
            volume_text,
            default_slice,
            gr.update(maximum=results['num_slices'] - 1, value=default_slice)
        )

    except Exception as e:
        return None, f"分析失败: {str(e)}", 0, gr.update(maximum=0)


def get_slice_image(slice_idx):
    """获取指定切片的可视化图像"""
    global current_results

    if current_results is None:
        return None

    slice_idx = int(slice_idx)
    if slice_idx < 0 or slice_idx >= current_results['num_slices']:
        return None

    ct_slice = current_results['ct_data'][:, :, slice_idx]
    pred_slice = current_results['predictions'][:, :, :, slice_idx]

    seg = initialize_model()
    img = seg.get_slice_visualization(ct_slice, pred_slice, alpha=0.5)

    # 转置以正确显示
    img = np.transpose(img, (1, 0, 2))

    return img


def update_slice(slice_idx):
    """更新切片显示"""
    return get_slice_image(slice_idx)


def generate_volume_report(results):
    """生成体积报告文本"""
    lines = []
    lines.append("## 肌肉体积分析报告")
    lines.append("")
    lines.append(f"**CT切片数量**: {results['num_slices']}")
    lines.append(f"**体素尺寸**: {results['voxel_dims'][0]:.2f} x {results['voxel_dims'][1]:.2f} x {results['voxel_dims'][2]:.2f} mm")
    lines.append("")
    lines.append("### 各肌肉体积")
    lines.append("")
    lines.append("| 肌肉名称 | 体积 (cm³) |")
    lines.append("|----------|-----------|")

    for muscle_name, data in results['volumes'].items():
        color_idx = list(MUSCLE_NAMES.keys()).index(muscle_name)
        color = MUSCLE_COLORS[color_idx]
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        lines.append(f"| <span style='color:{color_hex}'>{data['zh_name']}</span> | {data['volume_cm3']:.2f} |")

    lines.append("")
    lines.append(f"**总肌肉体积**: {results['total_volume_cm3']:.2f} cm³")

    # 左右对比
    lines.append("")
    lines.append("### 左右侧对比")
    lines.append("")
    lines.append("| 肌肉 | 左侧 (cm³) | 右侧 (cm³) | 差异 |")
    lines.append("|------|-----------|-----------|------|")

    muscle_pairs = [
        ('autochthon_left', 'autochthon_right', '背部肌群'),
        ('gluteus_maximus_left', 'gluteus_maximus_right', '臀大肌'),
        ('gluteus_medius_left', 'gluteus_medius_right', '臀中肌'),
        ('gluteus_minimus_left', 'gluteus_minimus_right', '臀小肌'),
        ('iliopsoas_left', 'iliopsoas_right', '髂腰肌'),
    ]

    for left, right, name in muscle_pairs:
        left_vol = results['volumes'][left]['volume_cm3']
        right_vol = results['volumes'][right]['volume_cm3']
        diff = left_vol - right_vol
        avg = (left_vol + right_vol) / 2
        diff_pct = (diff / avg * 100) if avg > 0 else 0
        lines.append(f"| {name} | {left_vol:.2f} | {right_vol:.2f} | {diff_pct:+.1f}% |")

    return "\n".join(lines)


def create_legend():
    """创建颜色图例"""
    legend_items = []
    for i, (name, info) in enumerate(MUSCLE_NAMES.items()):
        color = MUSCLE_COLORS[i]
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        legend_items.append(f"<span style='background-color:{color_hex};color:white;padding:2px 8px;margin:2px;border-radius:3px;display:inline-block;font-size:12px'>{info['zh']}</span>")
    return " ".join(legend_items)


# 创建Gradio界面
with gr.Blocks(title="肌肉分割可视化", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 肌肉分割可视化应用")
    gr.Markdown("上传CT图像文件 (.nii.gz)，查看肌肉分割结果和体积分析")

    with gr.Row():
        with gr.Column(scale=1):
            # 文件上传
            ct_input = gr.File(label="上传CT文件 (.nii.gz)", file_types=[".nii", ".nii.gz", ".gz"])
            analyze_btn = gr.Button("开始分析", variant="primary")

            # 切片滑块
            slice_slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                value=0,
                label="切片编号",
                interactive=True
            )

            # 颜色图例
            gr.Markdown("### 颜色图例")
            gr.HTML(create_legend())

        with gr.Column(scale=2):
            # 图像显示
            image_output = gr.Image(label="CT切片与肌肉分割", type="numpy")

    # 体积报告
    volume_output = gr.Markdown(label="体积分析报告")

    # 事件处理
    analyze_btn.click(
        fn=analyze_ct,
        inputs=[ct_input],
        outputs=[image_output, volume_output, slice_slider, slice_slider]
    )

    slice_slider.change(
        fn=update_slice,
        inputs=[slice_slider],
        outputs=[image_output]
    )


if __name__ == '__main__':
    print("启动肌肉分割可视化应用...")
    print("访问: http://localhost:7860")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
