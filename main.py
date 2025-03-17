# main.py
import torch
import torch.nn as nn
from data_loading import get_dataloaders
from models.diffusion import DiffusionModel
from train import train, val, save_checkpoint, load_checkpoint
from utils import app
import os
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
import time


torch.cuda.empty_cache()

# Hyperparameters
batch_size = 16
in_channel = 1
hidden_channel = 1
out_channel = 1
input_size = 252
hidden_size = 252
T = 5
learning_rate = 0.0005
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
criterion = nn.BCEWithLogitsLoss() 

torch.cuda.empty_cache()

# Data loading
image_dir = 'cropped_images'  # Path to cropped images
mask_dir = 'cropped_masks'    # Path to cropped masks
val_split = 0.05
test_split = 0.05
train_loader, val_loader, test_loader = get_dataloaders(
    image_dir, mask_dir, batch_size, val_split, test_split
)

# Model, loss, optimizer
model = DiffusionModel(
    batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size, T, num_classes
).to(device)
# criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Checkpoint 
checkpoint_path = "checkpoint\\5layers_checkpoint .pth"
start_epoch = 0
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
else:
    print(
        f"No checkpoint found at {checkpoint_path}, starting from epoch {start_epoch + 1}"
    )
 
# Training loop 
epoch_times = []
for epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()
    save_checkpoint(model, optimizer, epoch, checkpoint_path)
    train_loss = train(model, train_loader, optimizer, device, epoch, batch_size, checkpoint_path, criterion)    
    val_loss = val(model, val_loader, device, batch_size, criterion)
    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_times.append(epoch_time)
    save_checkpoint(model, optimizer, epoch, checkpoint_path)
    print(
        f"Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Epoch Time: {epoch_time:.2f} seconds"
    )
total_training_time = sum(epoch_times)
print(f"Total Training Time: {total_training_time:.2f} seconds")

if os.path.exists(checkpoint_path):  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

save_dir = "./imgs/5layers"
test = app(model, test_loader, device, batch_size, save_dir)
