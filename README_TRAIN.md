训练说明（U-Net 2D 切片）

文件简介：
- `scripts/train_unet.py`：基于 2D 轴向切片的 U-Net 训练脚本。它会遍历工作区中的 s0000..sNNNN 文件夹，读取 `ct.nii.gz` 以及 `segmentations/` 下的 per-structure NIfTI 掩码，并将每个轴向切片作为训练样本。

运行要求：已创建虚拟环境并安装 `requirements.txt` 中列出的包（或在项目虚拟环境中安装）。

示例运行：
在项目根目录（包含 s0000.. 文件夹）中运行：

```bash
python3 scripts/train_unet.py
```

脚本会默认进行一个非常小规模的运行（2 个 epoch），并把检查点和 `label_map.json` 保存在 `outputs/` 下。

建议：
- 数据量较小或内存有限时，降低 `target_shape` 和 `batch_size`。
- 若想训练 3D U-Net 或使用更复杂的模型（例如 UNeXt），可以把数据集类扩展成 3D patch 或用 PyTorch Lightning 封装训练循环。
