# utils.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from math import log10
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchmetrics import JaccardIndex, Dice, Precision, Recall, F1Score


def plot_heatmap(data, extent, vmin, vmax, cmap='jet'):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
    ax.set_xticks(np.linspace(extent[0], extent[1], 5))
    ax.set_yticks(np.linspace(extent[2], extent[3], 5))
    ax.set_xticklabels([f"{int(label)}°{'W' if label < 0 else 'E'}" for label in np.linspace(extent[0], extent[1], 5)])
    ax.set_yticklabels([f"{int(label)}°{'S' if label < 0 else 'N'}" for label in np.linspace(extent[2], extent[3], 5)])
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.04])
    ax.grid(True, linestyle='--', color='black', alpha=0.5)
    cbar = plt.colorbar(heatmap, cax=cax, orientation='horizontal', pad=-5)
    plt.show()

def calculate_iou(y_true, y_pred):
    y_pred = y_pred > -1
    if y_true.sum() == 0:
        return 1.0
    jaccard = JaccardIndex(task="binary").to("cuda")
    iou = jaccard(y_pred, y_true)
    return iou.item()

def calculate_dice(y_true, y_pred):
    y_pred = y_pred > 0
    dice_metric = Dice().to("cuda")
    dice = dice_metric(y_pred.int(), y_true.int())
    return dice


def app(model, dataloader, device, batch_size, save_dir):
    seg_loss = 0
    total_loss = 0
    total_iou = 0
    total_dice = 0

    # Load application data 
    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        if images.size(0) < batch_size:
            continue
        output_seg = model(images) 
        criterion = nn.BCEWithLogitsLoss()
        seg_loss = criterion(output_seg, masks)
        print(f"Batch Loss : {seg_loss.item()}")
        total_loss += seg_loss.item()

        for i in range(batch_size):
            # Calculate loss
            seg_loss = criterion(output_seg[i], masks[i])

            # Calculate IoU, Dice, Precision, Recall, F1 Score
            iou = calculate_iou(masks[i], output_seg[i])
            dice = calculate_dice(masks[i], output_seg[i])

            # Accumulate metrics
            total_iou += iou
            total_dice += dice

            # Visualization of predictions
            y_true_np = masks[i].squeeze(0).cpu().detach().numpy()
            y_pred_np = output_seg[i].squeeze(0).cpu().detach().numpy()
            x_np = images[i].squeeze(0).cpu().detach().numpy()
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(x_np)
            axes[0].set_title("Sample Image")
            axes[0].axis('off')  # Hide axes

            axes[1].imshow(y_true_np)
            axes[1].set_title("Sample Mask")
            axes[1].axis('off')  # Hide axes

            im = axes[2].imshow(y_pred_np, cmap='tab20b')
            axes[2].set_title("Predicted Mask")
            axes[2].axis('off')  # Hide axes
            fig.colorbar(im, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

            img_path = os.path.join(save_dir, f"result_{batch_idx * batch_size + i + 1}.png")
            plt.savefig(img_path, bbox_inches='tight', dpi=300)
            plt.close(fig) 
            torch.cuda.empty_cache()

    # Calculate average metrics
    num_samples = len(dataloader.dataset)
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_loss = total_loss / len(dataloader)
    print(f"Average IoU: {avg_iou}")
    print(f"Average Dice: {avg_dice}")
    print(f"Average Loss : {avg_loss}")
