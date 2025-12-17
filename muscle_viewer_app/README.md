# 肌肉分割可视化应用

基于训练好的U-Net模型，对CT图像进行肌肉分割并可视化。

## 功能

- 加载NIfTI格式的CT图像 (.nii.gz)
- 自动识别和分割10种肌肉
- 彩色叠加显示分割结果
- 计算各肌肉体积
- 左右侧肌肉对比分析

## 支持的肌肉类型

| 编号 | 肌肉名称 | 英文名 |
|-----|---------|--------|
| 1 | 左侧背部肌群 | Left Autochthon |
| 2 | 右侧背部肌群 | Right Autochthon |
| 3 | 左臀大肌 | Left Gluteus Maximus |
| 4 | 右臀大肌 | Right Gluteus Maximus |
| 5 | 左臀中肌 | Left Gluteus Medius |
| 6 | 右臀中肌 | Right Gluteus Medius |
| 7 | 左臀小肌 | Left Gluteus Minimus |
| 8 | 右臀小肌 | Right Gluteus Minimus |
| 9 | 左髂腰肌 | Left Iliopsoas |
| 10 | 右髂腰肌 | Right Iliopsoas |

## 安装依赖

```bash
pip install torch numpy nibabel scikit-image gradio tqdm
```

## 使用方法

### 方法1: 启动Web应用

```bash
python app.py
```

然后在浏览器中访问 http://localhost:7860

### 方法2: 命令行计算体积

```bash
python inference.py /path/to/ct.nii.gz
```

### 方法3: 在Python代码中调用

```python
from inference import MuscleSegmentor, calculate_volume_from_ct

# 简单调用
volumes, total = calculate_volume_from_ct('/path/to/ct.nii.gz')
print(f"总肌肉体积: {total:.2f} cm³")

# 完整调用
segmentor = MuscleSegmentor()
results = segmentor.segment_ct('/path/to/ct.nii.gz')

# 访问分割掩码
predictions = results['predictions']  # (10, H, W, D)

# 访问体积信息
for muscle, data in results['volumes'].items():
    print(f"{data['zh_name']}: {data['volume_cm3']:.2f} cm³")
```

## 文件结构

```
muscle_viewer_app/
├── app.py              # Gradio Web应用
├── inference.py        # 推理模块
├── README.md           # 本文件
└── models/
    ├── best_model.pth  # 训练好的模型权重
    └── label_map.json  # 标签映射文件
```

## 输出说明

- **CT图像显示**: 灰度CT切片
- **肌肉分割**: 彩色叠加在CT图像上，每种肌肉使用不同颜色
- **体积报告**: 包含各肌肉的体积（cm³）和左右侧对比

## 注意事项

- 输入CT图像应为NIfTI格式 (.nii.gz)
- 模型在256x256分辨率下进行推理，结果会自动缩放回原始尺寸
- GPU可用时会自动使用GPU加速
