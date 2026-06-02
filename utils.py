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
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import DiceScore


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
    y_pred = y_pred > 0
    if y_true.sum() == 0:
        return 1.0
    jaccard = JaccardIndex(task="binary").to(y_true.device)
    iou = jaccard(y_pred, y_true)
    return iou.item()

def calculate_dice(y_true, y_pred):
    y_pred = y_pred > 0
    dice_metric = DiceScore(num_classes=2).to(y_true.device)
    dice_metric.update(y_pred, y_true)
    dice = dice_metric.compute()
    return dice.item()

def calculate_proportion(y_pred):
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    mask_in_range = (y_pred_np > -1) & (y_pred_np < 1)
    proportion = np.mean(mask_in_range)
    return proportion


def app(model, dataloader, device, batch_size, save_dir, max_vis_samples=20):
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_proportion = 0
    vis_saved = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            if images.size(0) < batch_size:
                continue
            output_seg = model(images)

            total_loss += criterion(output_seg, masks).item()

            for i in range(batch_size):
                y_true = masks[i].unsqueeze(0)
                y_pred = output_seg[i].unsqueeze(0)

                total_iou  += calculate_iou(y_true, y_pred)
                total_dice += calculate_dice(y_true, y_pred)

                y_pred_np = output_seg[i].squeeze(0).cpu().detach().numpy()
                total_proportion += np.mean((y_pred_np > -1) & (y_pred_np < 1))

                # 只保存前 max_vis_samples 张可视化图，避免数万张 PNG 撑爆磁盘
                if vis_saved < max_vis_samples:
                    y_true_np = masks[i].squeeze(0).cpu().detach().numpy()
                    x_np      = images[i].squeeze(0).cpu().detach().numpy()

                    fig, axes = plt.subplots(1, 4, figsize=(26, 6),
                                             gridspec_kw={'width_ratios': [1, 1, 1.1, 1]})
                    axes[0].imshow(x_np);       axes[0].set_title("Input Image",       fontsize=35, pad=10); axes[0].axis('off')
                    axes[1].imshow(y_true_np);  axes[1].set_title("Ground Truth",      fontsize=35, pad=10); axes[1].axis('off')
                    im = axes[2].imshow(y_pred_np, cmap='tab20b', vmin=-10, vmax=2)
                    axes[2].set_title("Predicted Logits", fontsize=35, pad=10); axes[2].axis('off')
                    fig.colorbar(im, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04).ax.tick_params(labelsize=22)
                    binary_pred = (y_pred_np > 0).astype(float)
                    axes[3].imshow(binary_pred); axes[3].set_title("Binary Prediction", fontsize=35, pad=10); axes[3].axis('off')

                    img_path = os.path.join(save_dir, f"result_{batch_idx * batch_size + i + 1}.png")
                    plt.savefig(img_path, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    vis_saved += 1

            torch.cuda.empty_cache()

    # Calculate average metrics
    num_samples = len(dataloader.dataset)
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_loss = total_loss / len(dataloader)
    avg_proportion = total_proportion / num_samples

    print(f"Average IoU: {avg_iou}")
    print(f"Average Dice: {avg_dice}")
    print(f"Average Loss : {avg_loss}")
    print(f"Average proportion of pixels between -1 and 1: {avg_proportion}")

    # 🔴 修复 3: 返回计算指标，确保外部能写入 CSV 文件
    return {
        'loss': avg_loss,
        'iou': avg_iou,
        'dice': avg_dice,
        'proportion': avg_proportion
    }