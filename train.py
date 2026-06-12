import torch
import torch.nn as nn
import os
from utils import calculate_proportion, calculate_iou, calculate_dice


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


def val(model, dataloader, device, batch_size, criterion, checkpoint_path=None, thresh=0.5, verbose=True):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_proportion = 0.0
    num_samples = 0

    all_logits = []
    all_fg_logits = []
    all_bg_logits = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            # 跳过不完整的批次
            if images.size(0) < batch_size:
                continue

            output_seg = model(images)

            # Collect logit statistics for diagnosis
            logits_cpu = output_seg.cpu().float()
            masks_cpu = masks.cpu().float()
            all_logits.append(logits_cpu.flatten())
            fg_mask = masks_cpu > 0.5
            bg_mask = ~fg_mask
            if fg_mask.any():
                all_fg_logits.append(logits_cpu[fg_mask])
            if bg_mask.any():
                all_bg_logits.append(logits_cpu[bg_mask])

            # 计算损失
            seg_loss = criterion(output_seg, masks)
            total_loss += seg_loss.item()

            # 计算指标
            batch_size_current = images.size(0)
            num_samples += batch_size_current

            # 对每个样本单独计算指标
            for i in range(batch_size_current):
                y_true = masks[i].unsqueeze(0)  # 保持批次维度
                y_pred = output_seg[i].unsqueeze(0)

                # 计算IoU
                iou = calculate_iou(y_true, y_pred, thresh=thresh)
                total_iou += iou

                # 计算Dice
                dice = calculate_dice(y_true, y_pred, thresh=thresh)
                total_dice += dice

                # 计算proportion
                proportion = calculate_proportion(y_pred)
                total_proportion += proportion

    # 计算平均指标
    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_proportion = total_proportion / num_samples if num_samples > 0 else 0

    if verbose:
        # Print logit distribution for threshold diagnosis
        if all_logits and all_fg_logits and all_bg_logits:
            all_logits_cat = torch.cat(all_logits)
            fg_cat = torch.cat(all_fg_logits)
            bg_cat = torch.cat(all_bg_logits)
            print(f"Logit stats: min={all_logits_cat.min():.3f}, max={all_logits_cat.max():.3f}, "
                  f"mean={all_logits_cat.mean():.3f}, median={all_logits_cat.median():.3f}")
            print(f"  FG logits: mean={fg_cat.mean():.3f}, median={fg_cat.median():.3f}")
            print(f"  BG logits: mean={bg_cat.mean():.3f}, median={bg_cat.median():.3f}")
            opt_thresh = ((fg_cat.mean() + bg_cat.mean()) / 2).item()
            print(f"  Adaptive threshold (FG/BG midpoint): {opt_thresh:.3f}")

        print(f"Validation Results:")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Dice: {avg_dice:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average proportion of pixels between -1 and 1: {avg_proportion:.4f}")

    return {
        'loss': avg_loss,
        'iou': avg_iou,
        'dice': avg_dice,
        'proportion': avg_proportion
    }


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path, batch_idx=None):
    # 确保目录存在
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "batch_idx": batch_idx
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