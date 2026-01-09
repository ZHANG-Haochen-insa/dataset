#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Muscle Fat Analysis Web Application

A web-based interface for analyzing intramuscular adipose tissue (IMAT) in CT images.
Features:
- Drag and drop CT file upload (.nii.gz)
- Automatic muscle and fat segmentation using trained models
- Visualization with dark muscle and light fat highlighting
- Quantitative analysis results display

Author: hzhang02
Date: 2025-01-08
"""

import os
import sys
import json
import tempfile
import numpy as np
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
import base64

import torch
import torch.nn as nn

# Try to import gradio
try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Installing...")
    os.system("pip install gradio -q")
    import gradio as gr

# ============================================================================
# 1. Model Definition (same as training)
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
# 2. Analyzer Class
# ============================================================================

class MuscleFatAnalyzer:
    """Muscle and Fat Analysis Engine"""

    def __init__(self, muscle_model_path: str, fat_model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load muscle model
        self.muscle_model = UNet2D(in_ch=1, out_ch=1, features=[32, 64, 128, 256])
        muscle_checkpoint = torch.load(muscle_model_path, map_location=self.device, weights_only=False)
        self.muscle_model.load_state_dict(muscle_checkpoint['model_state_dict'])
        self.muscle_model.to(self.device)
        self.muscle_model.eval()

        # Load fat model
        self.fat_model = UNet2D(in_ch=1, out_ch=2, features=[32, 64, 128, 256])
        fat_checkpoint = torch.load(fat_model_path, map_location=self.device, weights_only=False)
        self.fat_model.load_state_dict(fat_checkpoint['model_state_dict'])
        self.fat_model.to(self.device)
        self.fat_model.eval()

        self.target_shape = (256, 256)
        print("Models loaded successfully!")

    def preprocess_slice(self, slice_img: np.ndarray) -> torch.Tensor:
        lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
        slice_normalized = np.clip(slice_img, lo, hi)
        if hi - lo > 0:
            slice_normalized = (slice_normalized - lo) / (hi - lo)
        else:
            slice_normalized = np.zeros_like(slice_normalized)

        slice_resized = resize(slice_normalized, self.target_shape, order=1,
                               preserve_range=True, anti_aliasing=True)
        img_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float()
        return img_tensor.to(self.device)

    def predict(self, slice_img: np.ndarray, threshold: float = 0.5):
        original_shape = slice_img.shape
        img_tensor = self.preprocess_slice(slice_img)

        with torch.no_grad():
            muscle_pred = torch.sigmoid(self.muscle_model(img_tensor))
            muscle_mask = (muscle_pred > threshold).float()[0, 0].cpu().numpy()

            fat_pred = torch.sigmoid(self.fat_model(img_tensor))
            sat_mask = (fat_pred[0, 0] > threshold).float().cpu().numpy()
            vat_mask = (fat_pred[0, 1] > threshold).float().cpu().numpy()
            all_fat_mask = np.logical_or(sat_mask, vat_mask).astype(np.float32)

        # Resize back to original shape
        muscle_mask = resize(muscle_mask, original_shape, order=0, preserve_range=True, anti_aliasing=False)
        sat_mask = resize(sat_mask, original_shape, order=0, preserve_range=True, anti_aliasing=False)
        vat_mask = resize(vat_mask, original_shape, order=0, preserve_range=True, anti_aliasing=False)
        all_fat_mask = resize(all_fat_mask, original_shape, order=0, preserve_range=True, anti_aliasing=False)

        # Binarize
        muscle_mask = (muscle_mask > 0.5).astype(np.float32)
        sat_mask = (sat_mask > 0.5).astype(np.float32)
        vat_mask = (vat_mask > 0.5).astype(np.float32)
        all_fat_mask = (all_fat_mask > 0.5).astype(np.float32)

        # Calculate IMAT
        imat_mask = np.logical_and(all_fat_mask, muscle_mask).astype(np.float32)
        pure_muscle_mask = np.logical_and(muscle_mask, ~imat_mask.astype(bool)).astype(np.float32)

        # Calculate areas
        muscle_area = muscle_mask.sum()
        imat_area = imat_mask.sum()
        pure_muscle_area = pure_muscle_mask.sum()
        sat_area = sat_mask.sum()
        vat_area = vat_mask.sum()

        imat_ratio = (imat_area / muscle_area * 100) if muscle_area > 0 else 0
        pure_muscle_ratio = (pure_muscle_area / muscle_area * 100) if muscle_area > 0 else 0

        return {
            'muscle_mask': muscle_mask,
            'pure_muscle_mask': pure_muscle_mask,
            'imat_mask': imat_mask,
            'sat_mask': sat_mask,
            'vat_mask': vat_mask,
            'all_fat_mask': all_fat_mask,
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


# ============================================================================
# 3. Visualization Functions
# ============================================================================

def create_visualization(slice_img, result, slice_idx, total_slices):
    """Create visualization figure"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Normalize CT for display
    lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
    ct_display = np.clip(slice_img, lo, hi)
    ct_display = (ct_display - lo) / (hi - lo + 1e-8)

    stats = result['stats']

    # Row 1, Col 1: Original CT
    axes[0, 0].imshow(ct_display, cmap='gray')
    axes[0, 0].set_title('Original CT Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Row 1, Col 2: Muscle + IMAT overlay
    axes[0, 1].imshow(ct_display, cmap='gray')
    overlay = np.zeros((*slice_img.shape, 4))
    overlay[result['pure_muscle_mask'] > 0.5] = [0.6, 0.1, 0.1, 0.6]  # Dark red
    overlay[result['imat_mask'] > 0.5] = [1.0, 0.9, 0.2, 0.8]  # Bright yellow
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f'Muscle (Dark) + IMAT (Light)\nIMAT Ratio: {stats["imat_ratio_percent"]:.2f}%',
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Legend
    dark_patch = mpatches.Patch(color=(0.6, 0.1, 0.1, 0.6), label='Pure Muscle')
    light_patch = mpatches.Patch(color=(1.0, 0.9, 0.2, 0.8), label='IMAT')
    axes[0, 1].legend(handles=[dark_patch, light_patch], loc='upper right', fontsize=9)

    # Row 1, Col 3: All segmentations
    axes[0, 2].imshow(ct_display, cmap='gray')
    combined = np.zeros((*slice_img.shape, 4))
    combined[result['sat_mask'] > 0.5] = [0.2, 0.4, 0.8, 0.5]  # Blue
    combined[result['vat_mask'] > 0.5] = [0.2, 0.8, 0.8, 0.5]  # Cyan
    combined[result['pure_muscle_mask'] > 0.5] = [0.8, 0.2, 0.2, 0.5]  # Red
    combined[result['imat_mask'] > 0.5] = [1.0, 0.9, 0.2, 0.7]  # Yellow
    axes[0, 2].imshow(combined)
    axes[0, 2].set_title('All Segmentations', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    patches = [
        mpatches.Patch(color=(0.2, 0.4, 0.8, 0.5), label='SAT'),
        mpatches.Patch(color=(0.2, 0.8, 0.8, 0.5), label='VAT'),
        mpatches.Patch(color=(0.8, 0.2, 0.2, 0.5), label='Muscle'),
        mpatches.Patch(color=(1.0, 0.9, 0.2, 0.7), label='IMAT'),
    ]
    axes[0, 2].legend(handles=patches, loc='upper right', fontsize=8)

    # Row 2, Col 1: Muscle mask
    axes[1, 0].imshow(ct_display, cmap='gray')
    axes[1, 0].imshow(result['muscle_mask'], alpha=0.5, cmap='Reds')
    axes[1, 0].set_title(f'Muscle Region\nArea: {stats["muscle_area_pixels"]:,} px', fontsize=12)
    axes[1, 0].axis('off')

    # Row 2, Col 2: IMAT mask
    axes[1, 1].imshow(ct_display, cmap='gray')
    axes[1, 1].imshow(result['imat_mask'], alpha=0.7, cmap='YlOrRd')
    axes[1, 1].set_title(f'IMAT (Intramuscular Fat)\nArea: {stats["imat_area_pixels"]:,} px', fontsize=12)
    axes[1, 1].axis('off')

    # Row 2, Col 3: Pie chart
    if stats['muscle_area_pixels'] > 0:
        areas = [stats['pure_muscle_area_pixels'], stats['imat_area_pixels']]
        labels = [f'Pure Muscle\n({stats["pure_muscle_ratio_percent"]:.1f}%)',
                  f'IMAT\n({stats["imat_ratio_percent"]:.1f}%)']
        colors = ['#8B0000', '#FFD700']

        if sum(areas) > 0:
            wedges, texts, autotexts = axes[1, 2].pie(
                areas, labels=labels, colors=colors,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 1 else '',
                startangle=90, explode=(0, 0.05), shadow=True
            )
            axes[1, 2].set_title('Muscle Composition', fontsize=12, fontweight='bold')
    else:
        axes[1, 2].text(0.5, 0.5, 'No muscle detected', ha='center', va='center', fontsize=12)
        axes[1, 2].set_title('Muscle Composition', fontsize=12)

    fig.suptitle(f'Slice {slice_idx} / {total_slices} - IMAT Analysis Results',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def create_results_text(stats, slice_idx, total_slices):
    """Create formatted results text"""
    text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ANALYSIS RESULTS                          ║
║                  Slice {slice_idx} / {total_slices}                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  MUSCLE ANALYSIS                                             ║
║  ────────────────────────────────────────────                ║
║  Total Muscle Area:     {stats['muscle_area_pixels']:>10,} pixels              ║
║  Pure Muscle Area:      {stats['pure_muscle_area_pixels']:>10,} pixels ({stats['pure_muscle_ratio_percent']:>5.1f}%)    ║
║  IMAT Area:             {stats['imat_area_pixels']:>10,} pixels ({stats['imat_ratio_percent']:>5.2f}%)     ║
║                                                              ║
║  FAT TISSUE ANALYSIS                                         ║
║  ────────────────────────────────────────────                ║
║  Subcutaneous Fat (SAT): {stats['sat_area_pixels']:>9,} pixels               ║
║  Visceral Fat (VAT):     {stats['vat_area_pixels']:>9,} pixels               ║
║                                                              ║
║  KEY METRIC                                                  ║
║  ────────────────────────────────────────────                ║
║  ★ IMAT Ratio: {stats['imat_ratio_percent']:>6.2f}%                                   ║
║    (Fat infiltration within muscle tissue)                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    return text


# ============================================================================
# 4. Global Analyzer Instance
# ============================================================================

# Model paths
BASE_DIR = '/local/hzhang02/data/dataset'
MUSCLE_MODEL_PATH = os.path.join(BASE_DIR, 'outputs_abdominal_wall', 'best_abdominal_wall_model.pth')
FAT_MODEL_PATH = os.path.join(BASE_DIR, 'outputs_fat', 'best_fat_model.pth')

# Initialize analyzer
analyzer = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = MuscleFatAnalyzer(MUSCLE_MODEL_PATH, FAT_MODEL_PATH)
    return analyzer


# ============================================================================
# 5. Gradio Interface Functions
# ============================================================================

def load_ct_file(file):
    """Load CT file and return basic info"""
    if file is None:
        return None, None, "Please upload a CT file (.nii.gz)", gr.update(maximum=1, value=0)

    try:
        # Load NIfTI file
        img = nib.load(file.name)
        ct_data = img.get_fdata().astype(np.float32)
        depth = ct_data.shape[2]

        info_text = f"""
CT File Loaded Successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Filename: {os.path.basename(file.name)}
Dimensions: {ct_data.shape[0]} x {ct_data.shape[1]} x {ct_data.shape[2]}
Total Slices: {depth}
Data Type: {ct_data.dtype}
HU Range: [{ct_data.min():.0f}, {ct_data.max():.0f}]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the slider to select a slice for analysis.
"""
        # Return middle slice as default
        default_slice = depth // 2

        return ct_data, depth, info_text, gr.update(maximum=depth-1, value=default_slice, interactive=True)

    except Exception as e:
        return None, None, f"Error loading file: {str(e)}", gr.update(maximum=1, value=0)


def analyze_slice(ct_data, slice_idx, total_slices):
    """Analyze a specific slice"""
    if ct_data is None:
        return None, "Please upload a CT file first."

    try:
        slice_idx = int(slice_idx)
        total_slices = int(total_slices)

        # Get analyzer
        anal = get_analyzer()

        # Get slice
        slice_img = ct_data[:, :, slice_idx]

        # Run prediction
        result = anal.predict(slice_img)

        # Create visualization
        fig = create_visualization(slice_img, result, slice_idx, total_slices)

        # Create results text
        results_text = create_results_text(result['stats'], slice_idx, total_slices)

        return fig, results_text

    except Exception as e:
        return None, f"Error during analysis: {str(e)}"


def analyze_multiple_slices(ct_data, total_slices, progress=gr.Progress()):
    """Analyze multiple slices across the volume"""
    if ct_data is None:
        return None, "Please upload a CT file first."

    try:
        total_slices = int(total_slices)
        anal = get_analyzer()

        # Sample slices evenly
        slice_indices = np.linspace(total_slices * 0.2, total_slices * 0.8, 8, dtype=int)

        all_stats = []

        for i, z in enumerate(progress.tqdm(slice_indices, desc="Analyzing slices...")):
            slice_img = ct_data[:, :, z]
            result = anal.predict(slice_img)
            stats = result['stats']
            stats['slice_index'] = int(z)
            all_stats.append(stats)

        # Create summary figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        slices = [s['slice_index'] for s in all_stats]
        imat_ratios = [s['imat_ratio_percent'] for s in all_stats]
        muscle_areas = [s['muscle_area_pixels'] for s in all_stats]

        # IMAT ratio trend
        axes[0].plot(slices, imat_ratios, 'o-', linewidth=2, markersize=10, color='#FFD700')
        axes[0].fill_between(slices, imat_ratios, alpha=0.3, color='#FFD700')
        axes[0].axhline(y=np.mean(imat_ratios), color='red', linestyle='--',
                       label=f'Mean: {np.mean(imat_ratios):.2f}%')
        axes[0].set_xlabel('Slice Index', fontsize=12)
        axes[0].set_ylabel('IMAT Ratio (%)', fontsize=12)
        axes[0].set_title('IMAT Ratio Across Slices', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Muscle area trend
        axes[1].plot(slices, muscle_areas, 's-', linewidth=2, markersize=10, color='#8B0000')
        axes[1].fill_between(slices, muscle_areas, alpha=0.3, color='#8B0000')
        axes[1].set_xlabel('Slice Index', fontsize=12)
        axes[1].set_ylabel('Muscle Area (pixels)', fontsize=12)
        axes[1].set_title('Muscle Area Across Slices', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        # Summary text
        total_muscle = sum(s['muscle_area_pixels'] for s in all_stats)
        total_imat = sum(s['imat_area_pixels'] for s in all_stats)
        avg_imat = np.mean(imat_ratios)

        summary_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                 MULTI-SLICE ANALYSIS SUMMARY                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Slices Analyzed:        {len(all_stats):>5}                               ║
║  Slice Range:            {min(slices):>5} - {max(slices):<5}                        ║
║                                                              ║
║  AGGREGATE RESULTS                                           ║
║  ────────────────────────────────────────────                ║
║  Total Muscle Area:      {total_muscle:>10,} pixels              ║
║  Total IMAT Area:        {total_imat:>10,} pixels              ║
║                                                              ║
║  IMAT RATIO STATISTICS                                       ║
║  ────────────────────────────────────────────                ║
║  ★ Average IMAT Ratio:   {avg_imat:>8.2f}%                          ║
║    Minimum:              {min(imat_ratios):>8.2f}%                          ║
║    Maximum:              {max(imat_ratios):>8.2f}%                          ║
║    Std Deviation:        {np.std(imat_ratios):>8.2f}%                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Per-Slice Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for s in all_stats:
            summary_text += f"  Slice {s['slice_index']:>3}: Muscle={s['muscle_area_pixels']:>6,}px, IMAT={s['imat_area_pixels']:>5,}px ({s['imat_ratio_percent']:.2f}%)\n"

        return fig, summary_text

    except Exception as e:
        return None, f"Error during multi-slice analysis: {str(e)}"


# ============================================================================
# 6. Build Gradio Interface
# ============================================================================

def create_app():
    """Create and return the Gradio app"""

    custom_css = """
        .gradio-container {max-width: 1400px !important}
        .results-text {font-family: monospace; white-space: pre; font-size: 12px;}
    """
    custom_theme = gr.themes.Soft(
        primary_hue="red",
        secondary_hue="yellow",
    )

    with gr.Blocks(title="IMAT Analysis Tool") as app:

        # Header
        gr.Markdown("""
        # 🏥 Intramuscular Adipose Tissue (IMAT) Analysis Tool

        **Analyze muscle composition and fat infiltration in CT images using deep learning.**

        This tool uses trained U-Net models to segment:
        - **Abdominal Wall Muscles** (shown in dark red)
        - **Intramuscular Fat (IMAT)** (shown in bright yellow)
        - **Subcutaneous Fat (SAT)** and **Visceral Fat (VAT)**

        ---
        """)

        # State variables
        ct_data_state = gr.State(None)
        total_slices_state = gr.State(0)

        with gr.Row():
            # Left column: Upload and controls
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Step 1: Upload CT File")
                file_input = gr.File(
                    label="Drag & Drop CT File Here (.nii.gz)",
                    file_types=[".nii", ".nii.gz", ".gz"],
                    type="filepath"
                )

                file_info = gr.Textbox(
                    label="File Information",
                    lines=8,
                    interactive=False,
                    value="Waiting for CT file upload..."
                )

                gr.Markdown("### 🎚️ Step 2: Select Slice")
                slice_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=1,
                    value=0,
                    label="Slice Index",
                    interactive=False
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Single Slice", variant="primary", size="lg")
                    analyze_all_btn = gr.Button("📊 Analyze Multiple Slices", variant="secondary", size="lg")

            # Right column: Results
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Analysis Results")

                with gr.Tabs():
                    with gr.TabItem("🖼️ Visualization"):
                        result_plot = gr.Plot(label="Segmentation Results")

                    with gr.TabItem("📋 Quantitative Results"):
                        result_text = gr.Textbox(
                            label="Detailed Metrics",
                            lines=25,
                            interactive=False,
                            elem_classes=["results-text"]
                        )

        # Footer
        gr.Markdown("""
        ---
        ### 📖 How to Use
        1. **Upload** a CT scan file in NIfTI format (.nii.gz)
        2. **Select** a slice using the slider
        3. **Click** "Analyze Single Slice" for detailed analysis of one slice
        4. **Click** "Analyze Multiple Slices" for a comprehensive volume analysis

        ### 🔬 About IMAT
        **Intramuscular Adipose Tissue (IMAT)** refers to fat deposits within muscle tissue.
        Higher IMAT ratios are associated with:
        - Reduced muscle quality
        - Metabolic disorders
        - Aging-related sarcopenia

        ---
        *Powered by PyTorch & U-Net | Models trained on CT imaging data*
        """)

        # Event handlers
        file_input.change(
            fn=load_ct_file,
            inputs=[file_input],
            outputs=[ct_data_state, total_slices_state, file_info, slice_slider]
        )

        analyze_btn.click(
            fn=analyze_slice,
            inputs=[ct_data_state, slice_slider, total_slices_state],
            outputs=[result_plot, result_text]
        )

        analyze_all_btn.click(
            fn=analyze_multiple_slices,
            inputs=[ct_data_state, total_slices_state],
            outputs=[result_plot, result_text]
        )

    return app


# ============================================================================
# 7. Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IMAT Analysis Web Application")
    print("=" * 60)

    # Check model files
    if not os.path.exists(MUSCLE_MODEL_PATH):
        print(f"ERROR: Muscle model not found: {MUSCLE_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(FAT_MODEL_PATH):
        print(f"ERROR: Fat model not found: {FAT_MODEL_PATH}")
        sys.exit(1)

    print(f"Muscle model: {MUSCLE_MODEL_PATH}")
    print(f"Fat model: {FAT_MODEL_PATH}")
    print("=" * 60)

    # Pre-load analyzer
    print("Pre-loading models...")
    get_analyzer()

    # Create and launch app
    app = create_app()

    print("\nLaunching web application...")
    print("Access the app at: http://localhost:7860")
    print("=" * 60)

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
