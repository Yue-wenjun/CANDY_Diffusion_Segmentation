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

def calculate_iou(y_true, y_pred, thresh=0.5):
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    if union == 0:
        return 1.0   # both empty → perfect
    return (intersection / union).item()

def calculate_dice(y_true, y_pred, thresh=0.5):
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    intersection = (y_pred * y_true).sum()
    denom = y_pred.sum() + y_true.sum()
    if denom == 0:
        return 1.0
    return (2.0 * intersection / denom).item()

def calculate_proportion(y_pred):
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    mask_in_range = (y_pred_np > -1) & (y_pred_np < 1)
    proportion = np.mean(mask_in_range)
    return proportion


def center_crop_pair(logits, masks, size):
    """Crop both tensors to the central size×size window (last two dims).

    Same arithmetic as Zheng et al. 2024's narrow() cropping: offset (H-size)//2.
    No-op when size is None or the tensors are already no larger than size.
    """
    if size is None:
        return logits, masks
    H, W = logits.shape[-2], logits.shape[-1]
    if H <= size and W <= size:
        return logits, masks
    top, left = (H - size) // 2, (W - size) // 2
    return (logits[..., top:top + size, left:left + size],
            masks[..., top:top + size, left:left + size])


# Logit thresholds swept during evaluation. 0.0 == sigmoid(logit) > 0.5.
# Spans the old imbalanced-BCE regime (boundary ~-2) and the corrected regime
# (boundary ~0 once pos_weight compensates the imbalance), finer near 0.
EVAL_THRESHOLDS = [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def evaluate_segmentation(model, dataloader, device, batch_size,
                          criterion=None, save_dir=None, max_vis_samples=0,
                          thresholds=None, center_crop=None):
    """Single source of truth for the IoU/Dice metric, shared by val() and app().

    - IoU/Dice are FOREGROUND-ONLY: images with an empty GT mask are excluded
      entirely (not scored 0 or 1). Their spurious firing is reported separately
      as empty_fp_rate.
    - pooled_iou is the confusion-matrix IoU = TP/(TP+FP+FN) accumulated over
      ALL pixels of ALL images (empty-GT included). This is the metric of
      Zheng et al. 2024 (their test IoU=0.40); read it at threshold 0.0
      (logit 0 == prob 0.5) for a like-for-like comparison.
    - The threshold sweep is vectorised on-device (accumulators stay on GPU,
      synced once at the end) so calling it every epoch in val() is cheap.
    Returns raw per-threshold curves; callers decide which threshold to report.
    """
    if thresholds is None:
        thresholds = EVAL_THRESHOLDS
    model.eval()

    T = torch.tensor(thresholds, device=device).view(-1, 1, 1, 1)   # [K,1,1,1]
    K = len(thresholds)
    fg_iou_sum  = torch.zeros(K, device=device)
    fg_dice_sum = torch.zeros(K, device=device)
    empty_fp    = torch.zeros(K, device=device)
    pooled_inter = torch.zeros(K, device=device)
    pooled_pred  = torch.zeros(K, device=device)
    pooled_gt    = torch.zeros((), device=device)
    fg_logit_sum = torch.zeros((), device=device); fg_logit_n = torch.zeros((), device=device)
    bg_logit_sum = torch.zeros((), device=device); bg_logit_n = torch.zeros((), device=device)
    fg_count, empty_count = 0, 0
    total_loss, total_proportion = 0.0, 0.0
    vis_saved = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            if images.size(0) < batch_size:
                continue
            logits = model(images)                       # [B,1,H,W]
            # Zheng et al. 2024 protocol: score only the central region whose
            # receptive field is fully inside the tile. Images cropped too so
            # the visualisations stay aligned with what is scored.
            logits, masks = center_crop_pair(logits, masks, center_crop)
            images = center_crop_pair(images, images, center_crop)[0]
            if criterion is not None:
                total_loss += criterion(logits, masks).item()

            # Class-conditional logit means (for the adaptive-threshold diagnostic).
            fgm = masks > 0.5
            bgm = ~fgm
            fg_logit_sum += logits[fgm].sum(); fg_logit_n += fgm.sum()
            bg_logit_sum += logits[bgm].sum(); bg_logit_n += bgm.sum()

            # Per-image gt sums, moved to CPU once per batch to branch cheaply.
            gt_sums = masks.sum(dim=(1, 2, 3))
            nonempty = (gt_sums > 0).tolist()

            for i in range(batch_size):
                gt = masks[i]                            # [1,H,W]
                pred = (logits[i].unsqueeze(0) > T).float()          # [K,1,H,W]
                psum = pred.sum(dim=(1, 2, 3))                       # [K]
                inter = (pred * gt).sum(dim=(1, 2, 3))               # [K]
                pooled_inter += inter
                pooled_pred  += psum
                pooled_gt    += gt_sums[i]
                if nonempty[i]:
                    fg_count += 1
                    gt_sum = gt_sums[i]
                    union = psum + gt_sum - inter
                    denom = psum + gt_sum
                    fg_iou_sum  += torch.where(union > 0, inter / union, torch.ones_like(inter))
                    fg_dice_sum += torch.where(denom > 0, 2.0 * inter / denom, torch.ones_like(inter))
                else:
                    empty_count += 1
                    empty_fp += pred.flatten(1).any(dim=1).float()  # [K]

                y_pred_np = logits[i].squeeze(0).cpu().detach().numpy()
                total_proportion += float(np.mean((y_pred_np > -1) & (y_pred_np < 1)))

                if save_dir and vis_saved < max_vis_samples:
                    y_true_np = masks[i].squeeze(0).cpu().detach().numpy()
                    x_np      = images[i].squeeze(0).cpu().detach().numpy()
                    fig, axes = plt.subplots(1, 4, figsize=(26, 6),
                                             gridspec_kw={'width_ratios': [1, 1, 1.1, 1]})
                    axes[0].imshow(x_np);       axes[0].set_title("Input Image",       fontsize=35, pad=10); axes[0].axis('off')
                    axes[1].imshow(y_true_np);  axes[1].set_title("Ground Truth",      fontsize=35, pad=10); axes[1].axis('off')
                    im = axes[2].imshow(y_pred_np, cmap='tab20b', vmin=-10, vmax=2)
                    axes[2].set_title("Predicted Logits", fontsize=35, pad=10); axes[2].axis('off')
                    fig.colorbar(im, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04).ax.tick_params(labelsize=22)
                    binary_pred = (y_pred_np > 0.0).astype(float)
                    axes[3].imshow(binary_pred); axes[3].set_title("Binary Prediction", fontsize=35, pad=10); axes[3].axis('off')
                    img_path = os.path.join(save_dir, f"result_{batch_idx * batch_size + i + 1}.png")
                    plt.savefig(img_path, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    vis_saved += 1

            torch.cuda.empty_cache()

    n = fg_count + empty_count
    fg_iou  = {t: (fg_iou_sum[k]  / fg_count).item() if fg_count else float("nan")
               for k, t in enumerate(thresholds)}
    fg_dice = {t: (fg_dice_sum[k] / fg_count).item() if fg_count else float("nan")
               for k, t in enumerate(thresholds)}
    fp_rate = {t: (empty_fp[k] / empty_count).item() if empty_count else float("nan")
               for k, t in enumerate(thresholds)}
    pooled_union = pooled_pred + pooled_gt - pooled_inter
    pooled_iou = {t: (pooled_inter[k] / pooled_union[k]).item() if pooled_union[k] > 0 else float("nan")
                  for k, t in enumerate(thresholds)}
    fg_lm = (fg_logit_sum / fg_logit_n).item() if fg_logit_n > 0 else float("nan")
    bg_lm = (bg_logit_sum / bg_logit_n).item() if bg_logit_n > 0 else float("nan")

    return {
        'thresholds':   list(thresholds),
        'fg_iou':       fg_iou,
        'fg_dice':      fg_dice,
        'pooled_iou':   pooled_iou,
        'empty_fp_rate': fp_rate,
        'fg_count':     fg_count,
        'empty_count':  empty_count,
        'loss':         total_loss / len(dataloader) if (criterion is not None and len(dataloader)) else float("nan"),
        'proportion':   total_proportion / n if n else float("nan"),
        'fg_logit_mean': fg_lm,
        'bg_logit_mean': bg_lm,
        'adaptive_thresh': (fg_lm + bg_lm) / 2,   # nan-safe: nan if either mean undefined
    }


def pick_best_threshold(res):
    """Threshold maximising foreground IoU over the swept grid (val-side use).
    Ties broken toward the HIGHER threshold: same IoU but fewer false positives
    on empty-GT images (higher threshold ⇒ less over-prediction)."""
    thr = res['thresholds']
    if res['fg_count'] == 0:
        return float("nan")
    return max(thr, key=lambda t: (res['fg_iou'][t], t))


def app(model, dataloader, device, batch_size, save_dir, max_vis_samples=20, thresh=None,
        center_crop=None):
    """Test-time evaluation. `thresh` is the FROZEN threshold chosen on validation
    (no leakage). If None (e.g. legacy checkpoint), fall back to the best-on-test
    threshold, which is an optimistic diagnostic ceiling — logged as such."""
    res = evaluate_segmentation(model, dataloader, device, batch_size,
                                criterion=nn.BCEWithLogitsLoss(),
                                save_dir=save_dir, max_vis_samples=max_vis_samples,
                                center_crop=center_crop)

    if res['fg_count'] == 0:
        print("[WARN] no foreground images in this test set — IoU undefined")
        return {'loss': res['loss'], 'iou': float("nan"), 'dice': float("nan"),
                'pooled_iou_05': res['pooled_iou'].get(0.0, float("nan")),
                'proportion': res['proportion'], 'best_thresh': float("nan"),
                'iou_best_on_test': float("nan"), 'empty_fp_rate': float("nan")}

    best_on_test = pick_best_threshold(res)          # diagnostic ceiling (leaky)
    if thresh is None:
        report_t = best_on_test
        print("[WARN] no frozen val threshold given — reporting best-on-test "
              f"(t={report_t:+.1f}), which is an optimistic upper bound.")
    else:
        # Snap the frozen threshold to the nearest swept grid point.
        report_t = min(res['thresholds'], key=lambda t: abs(t - thresh))

    # Zheng et al. 2024 comparison metric: pooled confusion-matrix IoU at
    # prob 0.5 (logit 0.0). Their published test value is 0.40.
    pooled_05 = res['pooled_iou'].get(0.0, float("nan"))

    print(f"Average Loss : {res['loss']}")
    print(f"[FG-only, n={res['fg_count']}] report@t={report_t:+.1f}: "
          f"IoU={res['fg_iou'][report_t]:.4f}  Dice={res['fg_dice'][report_t]:.4f}  "
          f"empty-FP={res['empty_fp_rate'][report_t]:.3f}   "
          f"(best-on-test t={best_on_test:+.1f}, IoU={res['fg_iou'][best_on_test]:.4f})")
    print(f"[Zheng-comparable] pooled IoU@0.5 = {pooled_05:.4f}  "
          f"(paper U-Net baseline: 0.40; pooled@report_t={res['pooled_iou'][report_t]:.4f})")

    return {
        'loss':             res['loss'],
        'iou':              res['fg_iou'][report_t],   # FG-only at the FROZEN val threshold
        'dice':             res['fg_dice'][report_t],
        'pooled_iou_05':    pooled_05,                 # Zheng et al. 2024 metric
        'proportion':       res['proportion'],
        'best_thresh':      report_t,
        'iou_best_on_test': res['fg_iou'][best_on_test],
        'empty_fp_rate':    res['empty_fp_rate'][report_t],
    }