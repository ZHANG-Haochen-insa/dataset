import os
import numpy as np
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go


def load_volume(path):
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return data


def extract_mesh(volume, level=None):
    # If no level provided, pick a percentile-based threshold
    if level is None:
        level = np.percentile(volume, 50)
    verts, faces, normals, values = measure.marching_cubes(volume, level=level)
    return verts, faces


def make_plotly_mesh(verts, faces, out_path):
    x, y, z = verts.T
    i, j, k = faces.T

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightgray', opacity=1.0)

    fig = go.Figure(data=[mesh])
    fig.update_layout(scene=dict(aspectmode='data'))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)


def main():
    gz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 's0000', 'ct.nii.gz'))
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'ct_mesh.html'))

    print('Loading volume:', gz_path)
    vol = load_volume(gz_path)
    print('Volume shape:', vol.shape)

    print('Extracting mesh...')
    verts, faces = extract_mesh(vol, level=np.percentile(vol, 60))
    print('Vertices:', verts.shape, 'Faces:', faces.shape)

    print('Saving interactive HTML to:', out_path)
    make_plotly_mesh(verts, faces, out_path)
    print('Done.')


if __name__ == '__main__':
    main()
