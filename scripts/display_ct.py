import os
import gzip
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def load_nifti_gz(path):
    """Load a .nii.gz file and return the image data as a numpy array and the affine/header."""
    # nibabel can directly load .nii.gz, but keep gzip read as fallback
    try:
        img = nib.load(path)
        data = img.get_fdata()
        return data, img.affine, img.header
    except Exception:
        with gzip.open(path, 'rb') as f:
            img = nib.Nifti1Image.from_bytes(f.read())
            data = img.get_fdata()
            return data, img.affine, img.header


def window_image(img, lower_percentile=1, upper_percentile=99):
    """Window image using percentiles to reduce outliers."""
    lo = np.percentile(img, lower_percentile)
    hi = np.percentile(img, upper_percentile)
    img = np.clip(img, lo, hi)
    if hi - lo > 0:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return img


def make_three_view(data, out_path):
    """Create a 3-panel PNG with axial, coronal and sagittal central slices."""
    # If 4D, take the first volume
    if data.ndim == 4:
        data = data[..., 0]

    # Ensure shape is (X, Y, Z)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {data.shape}")

    # Choose central slices
    x, y, z = data.shape
    cx, cy, cz = x // 2, y // 2, z // 2

    axial = np.rot90(data[:, :, cz])
    coronal = np.rot90(data[:, cy, :])
    sagittal = np.rot90(data[cx, :, :])

    axial = window_image(axial)
    coronal = window_image(coronal)
    sagittal = window_image(sagittal)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(axial, cmap='gray', aspect='equal')
    axes[0].set_title(f'Axial (z={cz})')
    axes[1].imshow(coronal, cmap='gray', aspect='equal')
    axes[1].set_title(f'Coronal (y={cy})')
    axes[2].imshow(sagittal, cmap='gray', aspect='equal')
    axes[2].set_title(f'Sagittal (x={cx})')

    for ax in axes:
        ax.axis('off')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Path relative to repository root
    gz_path = os.path.join(os.path.dirname(__file__), '..', 's0000', 'ct.nii.gz')
    gz_path = os.path.abspath(gz_path)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'ct_slices.png'))

    print('Loading:', gz_path)
    data, affine, header = load_nifti_gz(gz_path)
    print('Image shape:', data.shape)

    make_three_view(data, out_path)
    print('Saved three-view image to:', out_path)


if __name__ == '__main__':
    main()
