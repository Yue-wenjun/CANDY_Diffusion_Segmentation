import os
# os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
# os.environ['GDAL_HTTP_MAX_RETRY'] = '10'       # Set max retries to 10
# os.environ['GDAL_HTTP_RETRY_DELAY'] = '0.5'    # Set retry delay to 0.5 seconds
import numpy as np
import rasterio
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import pad
import matplotlib.pyplot as plt



class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initialize the dataset
        :param image_dir: Path to the cropped image directory
        :param mask_dir: Path to the cropped mask directory
        :param transform: Optional image transformations
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))  # Get all image file names
        self.masks = sorted(os.listdir(mask_dir))  # Get all mask file names


    def __len__(self):
        """Return the size of the dataset"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a single sample
        :param idx: Sample index
        :return: Image and mask
        """
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        with rasterio.open(img_path) as src:
            image = src.read()  # Read multi-band image

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        with rasterio.open(mask_path) as src:
            mask = src.read()  # Read multi-band image

        # Convert mask to multi-class labels
        # Assuming 1 for rainband, 4 for land, and other values for background (0)
        mask = np.where(mask == 1, 1, mask)  # 1: Rainband
        # mask = np.where(mask == 4, 2, mask)  # 4: Land
        mask = np.where(mask != 1, 0, mask)  # All other values as Background

        # Convert to PyTorch tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Replace NaN values in image with 0
        image = torch.nan_to_num(image, nan=0.0)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image.float(), mask.float()

def get_dataloaders(image_dir, mask_dir, batch_size=4, val_split=0.15, test_split=0.05, shuffle=True):
    """
    Get training and validation DataLoader
    :param image_dir: Path to the cropped image directory
    :param mask_dir: Path to the cropped mask directory
    :param batch_size: Batch size
    :param val_split: Validation split ratio
    :param shuffle: Whether to shuffle the data
    :return: Training and validation DataLoader
    """
    # Define the dataset
    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir)

    # Split into training and validation sets
    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Define DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example usage
    image_dir = 'cropped_images'  # Path to cropped images
    mask_dir = 'cropped_masks'    # Path to cropped masks

    # Get DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(image_dir, mask_dir, batch_size=16, val_split=0.05, test_split=0.05)

    # Print dataset information
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # Check one batch of data
    for images, masks in train_loader:
        print(f"Image shape: {images.shape}")  # (batch_size, channels, height, width)
        print(f"Mask shape: {masks.shape}")    # (batch_size, height, width)
        print(f"Unique mask classes: {torch.unique(masks)}")  # Print unique classes in the mask
        
        # Display the first image and its corresponding mask
        img = images[0].squeeze(0).numpy()
        mask = masks[0].squeeze(0).numpy()

        # Plot the image and mask side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title("Sample Image")
        axes[0].axis('off')  # Hide axes

        axes[1].imshow(mask, cmap='tab20b')  # Using a colormap for better visualization
        axes[1].set_title("Sample Mask")
        axes[1].axis('off')  # Hide axes

        plt.show()
        break