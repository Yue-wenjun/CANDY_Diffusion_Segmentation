# training.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import torch.nn.functional as F


def train(model, dataloader, optimizer, device, epoch, batch_size, checkpoint_path, criterion):
    model.train()
    running_loss = 0.0
    batch_loss = 0.0
    early_stop_counter = 0
    early_stop_threshold = 0.002
    patience = 5

    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        if images.size(0) < batch_size:
            continue
        if torch.isnan(images).any():
            print(f"NaN detected in images at batch {batch_idx}")
        if torch.isnan(masks).any():
            print(f"NaN detected in masks at batch {batch_idx}")
        optimizer.zero_grad()
        output_seg = model(images)
        loss = criterion(output_seg, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_loss += loss.item()

        if batch_idx % 10 == 9:
            print(
                f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}/{len(dataloader)}], 10 batch avg Loss: {batch_loss/10}, Loss: {loss.item()}"
            )
            batch_loss = 0
        if loss.item() < early_stop_threshold:
            early_stop_counter += 1
        else:
            early_stop_counter = 0
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}, batch {batch_idx}")
            break

        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss}")
    return avg_loss


def val(model, dataloader, device, batch_size, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            if images.size(0) < batch_size:
                continue
            output_seg = model(images)
            criterion = nn.BCEWithLogitsLoss()
            seg_loss = criterion(output_seg.squeeze(1), masks.squeeze(1))
            total_loss = seg_loss
            running_loss += total_loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Val Average Loss: {avg_loss}")
    return avg_loss


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found. Starting from epoch 0.")
        return 0
