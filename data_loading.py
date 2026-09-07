import os
import numpy as np
import rasterio
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import pad
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

def normalize_sar(img, valid=None):
    """Per-image SAR preprocessing, shared by ALL splits (train/val/test) so the
    model never sees a different input distribution at test time.

    SAR amplitudes are log-normal with a huge dynamic range; log1p compresses it,
    then per-image z-score centres it. Per-image stats are self-contained — no
    train-set statistics, hence no leakage and identical behaviour everywhere.

    `valid` (bool tensor, same shape) marks real-data pixels. No-data / no-image
    regions (image NaN) must be excluded from the mean/std — otherwise the large
    zero-filled no-data area (30-100% of some tiles) drags the statistics and
    mis-normalizes the actual sea/rainband signal. No-data pixels are set to 0
    after normalization.
    """
    img = torch.log1p(img.clamp(min=0))          # log(1+x); values are already >= 0
    if valid is None:
        return (img - img.mean()) / (img.std() + 1e-8)
    v = img[valid]
    if v.numel() == 0:
        return torch.zeros_like(img)             # fully no-data tile → all zeros
    out = (img - v.mean()) / (v.std() + 1e-8)
    return out * valid                           # no-data → 0


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, indices=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        all_images = sorted(os.listdir(image_dir))
        all_masks = sorted(os.listdir(mask_dir))

        # Pairing is positional (images[i] ↔ masks[i]); a count mismatch means the
        # sorted lists cannot align — fail loud rather than train on mispaired data.
        if len(all_images) != len(all_masks):
            raise ValueError(
                f"image/mask count mismatch: {len(all_images)} images vs "
                f"{len(all_masks)} masks in '{image_dir}' / '{mask_dir}'"
            )
        if all_images:
            # ASCII only: Windows GBK consoles cannot encode fancy arrows.
            print(f"[pairing] {len(all_images)} pairs; "
                  f"first={all_images[0]}<->{all_masks[0]}  last={all_images[-1]}<->{all_masks[-1]}")

        if indices is not None:
            self.images = [all_images[i] for i in indices]
            self.masks = [all_masks[i] for i in indices]
        else:
            self.images = all_images
            self.masks = all_masks

        num_samples = len(self.images)
        print(f"检测到豪华内存！正在构建 {num_samples} 个样本的终极巨型 Tensor...")

        # 提前在内存中开辟一整块连续的连续空间，彻底消灭运行时的 stack 开销！
        # 假设你的图像尺寸是 252x252，单通道
        self.all_images = torch.zeros((num_samples, 1, 252, 252), dtype=torch.float32)
        self.all_masks = torch.zeros((num_samples, 1, 252, 252), dtype=torch.float32)

        milestones = {int(num_samples * p) for p in (0.25, 0.5, 0.75)}
        for i in range(num_samples):
            if i in milestones:
                print(f"  {i}/{num_samples} ({100*i//num_samples}%)");
            img_path = os.path.join(self.image_dir, self.images[i])
            with rasterio.open(img_path) as src:
                image = src.read()

            mask_path = os.path.join(self.mask_dir, self.masks[i])
            with rasterio.open(mask_path) as src:
                mask = src.read()

            # Two distinct ignore regions, both encoded as -1 in the mask:
            #   land       — mask value 4 (a distinct annotated class; rainbands
            #                exist only on the sea, and land patches would score as
            #                false positives — matches Zheng et al.'s no-land eval)
            #   no-data    — image is NaN (tile extends beyond the SAR footprint;
            #                "no image"). Was zero-filled and scored as sea, and
            #                polluted the per-image normalization. Now ignored.
            # Label: rainband(1) → 1, land/no-data → -1 IGNORE, else (sea) → 0.
            nodata = ~np.isfinite(image)                     # [1,H,W] bool, "no image"
            land = np.isclose(mask, 4.0)
            mask = np.where(mask == 1, 1.0, 0.0).astype(np.float32)
            mask[land] = -1.0
            mask[nodata] = -1.0

            img_raw = torch.nan_to_num(torch.from_numpy(image), nan=0.0).float()
            valid = torch.from_numpy(~nodata)
            # Normalize ONCE at load over real-data pixels; store the result so
            # __getitem__ is a pure memory read (no per-access recompute).
            self.all_images[i] = normalize_sar(img_raw, valid=valid)
            self.all_masks[i] = torch.from_numpy(mask).float()

        print("巨型 Tensor 构建完毕！现在内存读取速度达到了理论极限。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are already normalized at load (valid-pixel stats), so this is a
        # pure memory read.
        image = self.all_images[idx]
        mask = self.all_masks[idx]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def get_dataloaders(image_dir, mask_dir, batch_size=4, val_split=0.1, test_split=0.15, shuffle=True):
    """
    Get training and validation DataLoader
    :param image_dir: Path to the cropped image directory
    :param mask_dir: Path to the cropped mask directory
    :param batch_size: Batch size
    :param val_split: Validation split ratio
    :param test_split: Test split ratio
    :param shuffle: Whether to shuffle the data
    :return: Training and validation DataLoader
    """
    # Define the dataset
    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir)

    # Split into training, validation and test sets
    total_size = len(dataset)
    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - val_size - test_size

    indices = list(range(total_size))
    # 测试集固定为最后一部分
    test_indices = indices[-test_size:]
    # 剩余部分用于训练和验证
    remaining_indices = indices[:-test_size]

    # 如果需要验证集也相对固定（例如剩余部分的最后一段）
    train_indices = remaining_indices[:train_size]
    val_indices = remaining_indices[train_size:]

    # 使用 Subset 创建数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Define DataLoader
    # 训练集：火力全开
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,  # 【关键】开启锁页内存，打通内存到显存的高速通道
        drop_last=True  # （可选）丢弃最后不够一个 Batch 的数据，防止 BatchNorm 报错
    )

    # 验证集：同样加速
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # 测试集：保持一致
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_test_only_dataloader(image_dir, mask_dir, batch_size=4, test_split=0.15):
    """
    只加载测试集（后 test_split 比例），不加载训练/验证数据到内存。
    相比 get_dataloaders，内存占用仅为 test_split 倍。
    """
    all_images = sorted(os.listdir(image_dir))
    total = len(all_images)
    test_size = int(test_split * total)
    test_indices = list(range(total))[-test_size:]

    print(f"仅加载测试集: 共 {total} 个样本，取后 {test_size} 个 ({test_split*100:.0f}%)")
    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, indices=test_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


class _AugSubset(torch.utils.data.Dataset):
    """Wraps a CustomDataset subset with optional random flip augmentation."""
    def __init__(self, base, indices, augment=False):
        self.base    = base
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img  = self.base.all_images[self.indices[i]]   # already normalized at load
        mask = self.base.all_masks[self.indices[i]]

        if self.augment:
            # Random 90° rotation (k=0,1,2,3)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img  = torch.rot90(img,  k, dims=[-2, -1])
                mask = torch.rot90(mask, k, dims=[-2, -1])
            # Horizontal flip
            if torch.rand(1).item() > 0.5:
                img  = torch.flip(img,  dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            # Vertical flip
            if torch.rand(1).item() > 0.5:
                img  = torch.flip(img,  dims=[-2])
                mask = torch.flip(mask, dims=[-2])
        return img, mask


def get_kfold_dataloaders(image_dir, mask_dir, batch_size=4, n_splits=4, fold=0,
                          shuffle=True, save_split_path=None, test_ratio=0.15):
    """
    Hold out the last test_ratio of data as a fixed test set (shared across all folds),
    then perform K-fold CV on the remaining (1 - test_ratio) portion.

    Split per fold (default test_ratio=0.15, n_splits=4):
        ~63.75% train  |  ~21.25% val  |  15% held-out test
    """
    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir)
    all_image_names = dataset.images
    total = len(dataset)

    # Fixed held-out test set: last test_ratio of all samples
    test_size = int(test_ratio * total)
    trainval_indices = list(range(total - test_size))   # first 85%

    print(f"Dataset split: {len(trainval_indices)} trainval / {test_size} held-out test "
          f"({test_ratio*100:.0f}%)")

    # K-fold on trainval only
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
    train_indices, val_indices = None, None
    for fold_idx, (tr_rel, va_rel) in enumerate(kfold.split(trainval_indices)):
        if fold_idx == fold:
            train_indices = [trainval_indices[i] for i in tr_rel]
            val_indices   = [trainval_indices[i] for i in va_rel]
            break

    print(f"Fold {fold}: {len(train_indices)} train / {len(val_indices)} val")

    # Save val filenames for later test-set reconstruction
    val_filenames = [all_image_names[i] for i in val_indices]
    if save_split_path:
        os.makedirs(os.path.dirname(save_split_path), exist_ok=True)
        with open(save_split_path, 'w') as f:
            json.dump(val_filenames, f)
        print(f"Fold {fold} val split saved to: {save_split_path}")

    train_loader = DataLoader(
        _AugSubset(dataset, train_indices, augment=True),
        batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        _AugSubset(dataset, val_indices, augment=False),
        batch_size=batch_size, shuffle=False, pin_memory=True,
    )

    return train_loader, val_loader


def get_kfold_test_dataloader(image_dir, mask_dir, batch_size=4, test_ratio=0.15):
    """
    Returns the same fixed held-out test set used by get_kfold_dataloaders.
    Call this after k-fold training to evaluate on the shared test set.
    """
    all_images = sorted(os.listdir(image_dir))
    total = len(all_images)
    test_size = int(test_ratio * total)
    test_indices = list(range(total - test_size, total))

    print(f"Held-out test set: {test_size} samples ({test_ratio*100:.0f}% of {total})")
    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, indices=test_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# 保留原有的测试代码
if __name__ == '__main__':
    # Example usage
    image_dir = 'cropped_images'  # Path to cropped images
    mask_dir = 'cropped_masks'  # Path to cropped masks

    # Get DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(image_dir, mask_dir, batch_size=16, val_split=0.05,
                                                            test_split=0.05)

    # Print dataset information
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # Check one batch of data
    for images, masks in train_loader:
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Unique mask classes: {torch.unique(masks)}")
        break