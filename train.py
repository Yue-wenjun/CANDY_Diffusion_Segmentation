import torch
import torch.nn as nn
import os
from utils import evaluate_segmentation, pick_best_threshold


def train(model, dataloader, optimizer,scheduler, device, epoch, batch_size, checkpoint_path, criterion, save_interval=100):
    model.train()
    running_loss = 0.0
    batch_loss = 0.0
    early_stop_counter = 0
    early_stop_threshold = 0.0002
    patience = 5

    train_fg_logits = []
    train_bg_logits = []

    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        # 检查输入数据
        if torch.isnan(images).any() or torch.isnan(masks).any():
            print(f"NaN detected in input data at batch {batch_idx}")
            continue

        if images.size(0) < batch_size:
            continue

        optimizer.zero_grad()
        output_seg = model(images)

        # 检查模型输出
        if torch.isnan(output_seg).any():
            print(f"NaN detected in model output at batch {batch_idx}")
            continue

        loss = criterion(output_seg, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss at batch {batch_idx}, skipping...")
            continue

        loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 检查梯度
        nan_grads = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                nan_grads = True
                break

        if nan_grads:
            print(f"NaN/Inf gradients at batch {batch_idx}, skipping update...")
            optimizer.zero_grad()  # 清除有问题的梯度
            continue

        optimizer.step()
        scheduler.step()

        # Collect train logit stats (sampled every 20 batches to stay cheap)
        if batch_idx % 20 == 0:
            with torch.no_grad():
                lc = output_seg.detach().cpu().float()
                mc = masks.detach().cpu().float()
                fg = mc > 0.5
                bg = ~fg
                if fg.any():
                    train_fg_logits.append(lc[fg].mean().item())
                if bg.any():
                    train_bg_logits.append(lc[bg].mean().item())

        running_loss += loss.item()
        batch_loss += loss.item()

        # 定期保存检查点
        if batch_idx % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}")

        if batch_idx % 10 == 9:
            print(
                f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}/{len(dataloader)}], 10 batch avg Loss: {batch_loss / 10}, Loss: {loss.item()}")
            batch_loss = 0.0

        # if loss.item() < early_stop_threshold:
        #     early_stop_counter += 1
        # else:
        #     early_stop_counter = 0
        #
        # if early_stop_counter >= patience:
        #     print(f"Early stopping triggered at epoch {epoch}, batch {batch_idx}")
        #     # 保存最终检查点
        #     save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path, batch_idx)
        #     break

    avg_loss = running_loss / len(dataloader)
    if train_fg_logits and train_bg_logits:
        print(f"Train logit stats: FG mean={sum(train_fg_logits)/len(train_fg_logits):.3f}, "
              f"BG mean={sum(train_bg_logits)/len(train_bg_logits):.3f}")
    print(f"Epoch [{epoch}] Average Loss: {avg_loss}")

    # 保存每个epoch结束时的检查点
    save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path, "final")

    return avg_loss


def val(model, dataloader, device, batch_size, criterion, checkpoint_path=None, thresh=None, verbose=True):
    # Delegates to the shared metric so val and test are guaranteed identical:
    # foreground-only IoU/Dice, swept thresholds. The best-IoU threshold on THIS
    # validation set is returned as 'best_thresh' and later frozen for test.
    res = evaluate_segmentation(model, dataloader, device, batch_size, criterion=criterion)

    best_t   = pick_best_threshold(res)
    if res['fg_count'] == 0:
        avg_iou = avg_dice = 0.0
        best_t = 0.0
    else:
        avg_iou  = res['fg_iou'][best_t]
        avg_dice = res['fg_dice'][best_t]

    if verbose:
        print(f"  FG logit mean={res['fg_logit_mean']:.3f}  BG logit mean={res['bg_logit_mean']:.3f}  "
              f"adaptive(midpoint)={res['adaptive_thresh']:.3f}")
        print(f"Validation [FG-only, n={res['fg_count']}]: "
              f"IoU={avg_iou:.4f}  Dice={avg_dice:.4f}  Loss={res['loss']:.4f}  "
              f"best_thresh={best_t:+.1f}  empty-FP={res['empty_fp_rate'].get(best_t, float('nan')):.3f}")

    return {
        'loss': res['loss'],
        'iou': avg_iou,            # FG-only IoU at the val-optimal threshold
        'dice': avg_dice,
        'proportion': res['proportion'],
        'best_thresh': best_t,     # frozen and applied at test time (no leakage)
    }


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path, batch_idx=None, best_thresh=None):
    # 确保目录存在
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "batch_idx": batch_idx,
        # Val-selected decision threshold, frozen here and reused at test time.
        "best_thresh": best_thresh,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # 【新增】保存 scheduler 状态
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 【新增】加载 scheduler 状态
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif scheduler is not None:
            print("警告: 之前的 Checkpoint 中没有 Scheduler 状态，学习率将重新开始调度。")

        start_epoch = checkpoint['epoch']
        # start_epoch = 1  <--- 【已修复】这行强制把读取的 epoch 变成了 1，会导致逻辑错误，已注释
        batch_idx = checkpoint.get("batch_idx", 0)

        print(f"Resuming training from epoch {start_epoch}, batch {batch_idx}")
        # 【已修复】统一返回两个值，避免外层调用时解包报错
        return start_epoch, batch_idx
    else:
        print("No checkpoint found. Starting from epoch 0.")
        return 0, 0