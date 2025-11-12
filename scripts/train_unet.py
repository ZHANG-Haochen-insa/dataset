import os
import json
import glob
import random
from typing import List

import numpy as np
import nibabel as nib
from skimage.transform import resize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


def find_subjects(root: str) -> List[str]:
    # subjects are folders starting with s
    paths = sorted(glob.glob(os.path.join(root, 's*')))
    return [p for p in paths if os.path.isdir(p)]


def build_label_map(subjects: List[str], seg_subfolder='segmentations'):
    # collect all segmentation filenames across subjects
    names = set()
    for s in subjects:
        segdir = os.path.join(s, seg_subfolder)
        if os.path.isdir(segdir):
            for p in glob.glob(os.path.join(segdir, '*.nii*')):
                names.add(os.path.basename(p))
    names = sorted(names)
    # map name -> channel index
    label_map = {name: idx for idx, name in enumerate(names)}
    return label_map


class SliceDataset(Dataset):
    """2D axial-slice dataset from 3D CT and per-structure segmentation files.

    Produces (image, mask) where image is (1,H,W) and mask is (C,H,W) multi-channel binary masks.
    """
    def __init__(self, subjects: List[str], label_map: dict, seg_subfolder='segmentations', transform=None, target_shape=(256,256)):
        self.items = []
        self.label_map = label_map
        self.transform = transform
        self.target_shape = target_shape

        for s in subjects:
            ct_path = os.path.join(s, 'ct.nii.gz')
            segdir = os.path.join(s, seg_subfolder)
            if not os.path.exists(ct_path):
                continue
            # find seg files for this subject
            segfiles = {}
            if os.path.isdir(segdir):
                for p in glob.glob(os.path.join(segdir, '*.nii*')):
                    name = os.path.basename(p)
                    if name in label_map:
                        segfiles[label_map[name]] = p

            img = nib.load(ct_path)
            data = img.get_fdata().astype(np.float32)
            depth = data.shape[2]
            for z in range(depth):
                self.items.append((ct_path, segfiles, z))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ct_path, segfiles, z = self.items[idx]
        img = nib.load(ct_path).get_fdata().astype(np.float32)
        slice_img = img[:, :, z]
        # normalize window using percentiles
        lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
        slice_img = np.clip(slice_img, lo, hi)
        if hi - lo > 0:
            slice_img = (slice_img - lo) / (hi - lo)
        else:
            slice_img = np.zeros_like(slice_img)

        H, W = self.target_shape
        slice_img = resize(slice_img, (H, W), order=1, preserve_range=True, anti_aliasing=True)
        # build multi-channel mask
        C = len(self.label_map)
        mask = np.zeros((C, H, W), dtype=np.float32)
        for ch, p in segfiles.items():
            m = nib.load(p).get_fdata().astype(np.float32)
            m_slice = m[:, :, z]
            m_slice = resize(m_slice, (H, W), order=0, preserve_range=True, anti_aliasing=False)
            mask[ch] = (m_slice > 0.5).astype(np.float32)

        # convert to tensors
        img_t = torch.from_numpy(slice_img).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).float()
        return img_t, mask_t


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
    def __init__(self, in_ch=1, out_ch=1, features=[32,64,128,256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # up path
        rev = list(reversed(features))
        up_in = features[-1]*2
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
            conv = self.ups[idx+1]
            x = trans(x)
            skip = skips[-(idx//2)-1]
            if x.shape != skip.shape:
                # center crop skip to x
                _,_,h,w = x.shape
                skip = skip[:, :, :h, :w]
            x = torch.cat([skip, x], dim=1)
            x = conv(x)
        return self.final(x)


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    # pred & target shape: (N, C, H, W) with pred probabilities {0,1}
    N, C = pred.shape[:2]
    pred = pred.view(N, C, -1)
    target = target.view(N, C, -1)
    inter = (pred * target).sum(-1)
    union = pred.sum(-1) + target.sum(-1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def train_loop(root='.', out_dir='outputs', epochs=10, batch_size=8, lr=1e-3, target_shape=(256,256)):
    subjects = find_subjects(root)
    if not subjects:
        raise RuntimeError('No subjects found')
    label_map = build_label_map(subjects)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

    random.shuffle(subjects)
    n = len(subjects)
    ntrain = max(1, int(n * 0.8))
    train_subs = subjects[:ntrain]
    val_subs = subjects[ntrain:]

    train_ds = SliceDataset(train_subs, label_map, target_shape=target_shape)
    val_ds = SliceDataset(val_subs, label_map, target_shape=target_shape)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D(in_ch=1, out_ch=len(label_map)).to(device)
    opt = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
        avg_loss = running_loss / (len(train_loader) + 1e-8)

        # validation
        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                preds = torch.sigmoid(model(imgs))
                preds_bin = (preds > 0.5).float()
                dices.append(dice_score(preds_bin, masks))
        val_dice = float(np.mean(dices)) if dices else 0.0

        print(f'Epoch {epoch}/{epochs}  train_loss={avg_loss:.4f}  val_dice={val_dice:.4f}')
        # save checkpoint
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()},
                   os.path.join(out_dir, f'checkpoint_epoch{epoch}.pth'))


if __name__ == '__main__':
    # basic entry
    print('Train U-Net (2D slices) script')
    print('This script expects dataset folders s0000.. with ct.nii.gz and segmentations/*.nii.gz')
    # default small run
    train_loop(root='.', out_dir='outputs', epochs=2, batch_size=8, lr=1e-3, target_shape=(256,256))
