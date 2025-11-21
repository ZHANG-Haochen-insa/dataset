# 3D Medical Image Segmentation - Training and Analysis

This directory contains Jupyter notebooks for training and analyzing 3D medical image segmentation models on CT scans.

## 📋 Contents

- `train_unet.ipynb` - U-Net model training notebook
- `segmentation_detection_analysis.ipynb` - Model evaluation and visualization notebook

## 🎯 Overview

This project implements a 2D U-Net model for multi-organ segmentation from CT scans. The model can segment 117 different anatomical structures including:
- Major organs (brain, heart, liver, kidneys, lungs, etc.)
- Skeletal structures (vertebrae, ribs, etc.)
- Vascular system (aorta, arteries, veins, etc.)
- Muscle groups

## 🔧 Requirements

### System Requirements
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

### Python Dependencies
All dependencies are listed in `../requirements.txt`:
```
numpy
nibabel
scikit-image
torch
matplotlib
plotly
pandas
tqdm
scipy
jupyter
jupyterlab
ipywidgets
```

## 📦 Installation

1. **Install dependencies** (if not already installed):
```bash
cd /home/hzhang02/dataset
pip3 install -r requirements.txt
```

2. **Configure Jupyter kernel**:
```bash
python3 -m ipykernel install --user --name=claude_env --display-name="Python 3 (claude_env)"
```

3. **Verify installation**:
```bash
python3 -c "import numpy, torch, nibabel, plotly; print('All packages installed successfully!')"
```

## 🚀 Quick Start

### Step 1: Train the Model

1. Launch JupyterLab:
```bash
cd /home/hzhang02/dataset/scripts
jupyter lab train_unet.ipynb
```

2. Select kernel: **"Python 3 (claude_env)"**

3. Run all cells to:
   - Load and prepare the dataset
   - Build the U-Net model
   - Train for specified epochs
   - Save checkpoints to `../outputs/`

**Training Configuration** (adjustable in Section 6):
- `EPOCHS = 5` - Number of training epochs
- `BATCH_SIZE = 8` - Batch size for training
- `LEARNING_RATE = 1e-3` - Learning rate
- `TARGET_SHAPE = (256, 256)` - Image size

**Expected Output**:
- `outputs/label_map.json` - Mapping of anatomical structures to channels
- `outputs/checkpoint_epochX.pth` - Model checkpoints
- `outputs/training_history.json` - Training metrics
- `outputs/training_history.png` - Training curves visualization

### Step 2: Evaluate and Analyze

1. Launch the analysis notebook:
```bash
jupyter lab segmentation_detection_analysis.ipynb
```

2. Select kernel: **"Python 3 (claude_env)"**

3. Run all cells to:
   - Load trained model
   - Perform inference on test data
   - Calculate evaluation metrics
   - Generate visualizations

**Evaluation Metrics**:
- Dice Coefficient
- IoU (Intersection over Union)
- Success Rate (at different thresholds)
- Per-structure performance analysis

**Generated Outputs**:
- `outputs/evaluation_results_*.csv` - Detailed metrics per structure
- `outputs/metrics_distribution_*.png` - Metric distributions
- `outputs/structure_ranking_*.png` - Performance rankings
- `outputs/segmentation_visualization_*.png` - 2D segmentation results
- `outputs/3d_*.html` - Interactive 3D visualizations

## 📊 Dataset Structure

Expected directory structure:
```
/home/hzhang02/dataset/
├── s0000/
│   ├── ct.nii.gz                    # CT scan
│   └── segmentations/               # Ground truth masks
│       ├── liver.nii.gz
│       ├── heart.nii.gz
│       ├── kidney_left.nii.gz
│       └── ... (117 structures)
├── s0001/
├── s0002/
└── ...
```

## 📈 Typical Workflow

1. **Initial Training** (2-5 epochs for testing):
```bash
# Run train_unet.ipynb with EPOCHS=2
```

2. **Quick Evaluation**:
```bash
# Run segmentation_detection_analysis.ipynb
```

3. **Full Training** (if results are promising):
```bash
# Increase EPOCHS to 20-50 in train_unet.ipynb
```

4. **Comprehensive Analysis**:
```bash
# Re-run segmentation_detection_analysis.ipynb with best checkpoint
```

## 🎨 Visualization Examples

The notebooks generate various visualizations:

### Training Progress
- Loss curves over epochs
- Validation Dice score trends
- Sample predictions vs. ground truth

### Evaluation Results
- 2D slice comparisons (CT + overlay)
- 3D mesh renderings (interactive HTML)
- Performance distribution histograms
- Structure ranking charts

## 🔍 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"
**Solution**: Make sure to select the correct kernel:
- In Jupyter: Kernel → Change Kernel → "Python 3 (claude_env)"

### Issue: CUDA out of memory
**Solution**: Reduce batch size in training notebook:
```python
BATCH_SIZE = 4  # or even 2
```

### Issue: Training is too slow
**Solution**:
- Reduce image size: `TARGET_SHAPE = (128, 128)`
- Reduce model complexity: `features=[16, 32, 64, 128]`
- Use fewer epochs for testing: `EPOCHS = 2`

### Issue: Low Dice scores
**Possible causes**:
- Insufficient training epochs (try 20-50)
- Learning rate too high/low (try 1e-4 or 5e-4)
- Dataset too small (consider data augmentation)

## 📚 Model Architecture

**U-Net 2D**:
- Encoder: 4 downsampling blocks [32, 64, 128, 256 features]
- Bottleneck: 512 features
- Decoder: 4 upsampling blocks with skip connections
- Output: 117 channels (one per anatomical structure)

**Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

**Optimizer**: Adam (lr=1e-3)

**Evaluation Metric**: Dice Coefficient

## 💡 Tips for Better Results

1. **Data Augmentation**: Consider adding:
   - Random rotation (±15°)
   - Random flipping (horizontal/vertical)
   - Elastic deformation
   - Intensity scaling

2. **Advanced Training**:
   - Learning rate scheduling (ReduceLROnPlateau)
   - Early stopping
   - Gradient clipping
   - Mixed precision training (for faster GPU training)

3. **Model Improvements**:
   - Try 3D U-Net instead of 2D
   - Use attention mechanisms (Attention U-Net)
   - Experiment with different loss functions (Dice Loss, Focal Loss)

4. **Ensemble Methods**:
   - Train multiple models with different random seeds
   - Average predictions for better results

## 📖 References

- **Dataset**: TotalSegmentator v2.0.1
- **Model**: U-Net (Ronneberger et al., 2015)
- **Framework**: PyTorch 2.9.0

## 📝 Notes

- Training time depends on GPU/CPU and dataset size
- First epoch is usually slower due to data loading
- Checkpoints are saved after each epoch (can be large files)
- Validation is performed after each training epoch

## 🤝 Contributing

To extend this project:
1. Add new anatomical structures to the label map
2. Implement 3D U-Net for better spatial context
3. Add more evaluation metrics (Hausdorff distance, surface distance)
4. Integrate with medical imaging viewers (3D Slicer, ITK-SNAP)

## 📄 License

This project is for educational and research purposes.

---

**Created**: November 2025
**Last Updated**: November 2025
