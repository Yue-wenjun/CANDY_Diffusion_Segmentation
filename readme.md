# CANDY Diffusion Model

This repository contains the implementation of the CANDY Diffusion Model, a novel diffusion-based framework for image segmentation and time-series prediction tasks.

## Ablation Study Design

We designed six ablation groups to analyze the contribution of each component in the model. Below are the details of each group and the corresponding code changes:

---

### 1. **Baseline**
- **Description**: The full CANDY Diffusion model with all components (CANDY modules, UNet decoder, Skip Connection, and Segmentation Head).
- **Code**: No changes are made to the original implementation.

---

### 2. **Ablation Group 1: Remove Skip Connection**
- **Description**: Remove Skip Connection to observe its impact on information propagation during the reverse diffusion process.
- **Code Changes**:
  - Modify the reverse diffusion process in the `forward` method:
    ```python
    # Original code with Skip Connection
    reverse_input = (1 - graph_factor) * input + graph_factor * origin[t]

    # Modified code without Skip Connection
    reverse_input = input  # Skip Connection removed
    ```

---

### 3. **Ablation Group 2: Replace CANDY Modules with Simple CNN Modules**
- **Description**: Replace the CANDY modules with simple CNN modules to analyze the effectiveness of the CANDY architecture.
- **Code Changes**:
  - Replace the `self.candies` initialization in the `__init__` method:
    ```python
    # Original CANDY modules
    self.candies = nn.ModuleList([
        CANDY(batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size)
        for _ in range(T)
    ])

    # Modified simple CNN modules
    self.candies = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )
        for _ in range(T)
    ])
    ```

---

### 4. **Ablation Group 3: Remove UNet Decoder**
- **Description**: Remove the UNet decoder and replace it with a simple linear decoder to evaluate its importance in the reverse diffusion process.
- **Code Changes**:
  - Replace the `self.unets` initialization in the `__init__` method:
    ```python
    # Original UNet decoder
    self.unets = nn.ModuleList([
        UNet(in_channel, out_channel)
        for _ in range(T)
    ])

    # Modified simple linear decoder
    self.unets = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.ReLU()
        )
        for _ in range(T)
    ])
    ```

---

### 5. **Ablation Group 4: Remove Segmentation Head**
- **Description**: Remove the Segmentation Head to observe the model's performance in unsupervised tasks.
- **Code Changes**:
  - Remove the `self.seg_head` and related code in the `forward` method:
    ```python
    # Original code with Segmentation Head
    output_seg = self.seg_head(output)
    output_seg = torch.sum(output_seg, dim=1, keepdim=True)
    return output_seg

    # Modified code without Segmentation Head
    return output  # Directly return the output of the reverse diffusion process
    ```

---

### 6. **Ablation Group 5: Replace ODE with SDE**
- **Description**: Replace the ODE structure with SDE (Stochastic Differential Equation) to analyze the impact of stochasticity on model performance.
- **Code Changes**:
  - Add random noise to the output of the CANDY modules in the forward diffusion process:
    ```python
    # Original ODE-based forward diffusion
    output = self.candies[t](input)
    origin[t] = input
    input = output

    # Modified SDE-based forward diffusion
    output = self.candies[t](input)
    noise = torch.randn_like(output) * 0.1  # Add random noise
    output = output + noise
    origin[t] = input
    input = output
    ```

---

### 7. **Ablation Group 6: Increase Diffusion Steps (T)**
- **Description**: Increase the number of diffusion steps (`T`) to evaluate the model's performance with more steps.
- **Code Changes**:
  - Modify the `T` value in the `__init__` method:
    ```python
    # Original T value
    self.T = 2  # Baseline

    # Modified T value
    self.T = 5  # Increased diffusion steps
    ```

---

## Code Structure

- `diffusionl.py`: Contains the implementation of the `DiffusionModel` class.
- `candy.py`: Implementation of the CANDY module.
- `unet.py`: Implementation of the UNet decoder.
- `train.py`: Training script for the model.
- `eval.py`: Evaluation script for the model.

---

## Experiment Setup

### Datasets
- **Time-Series Prediction Task**: Sea Surface Temperature (SST) dataset.
- **Image Segmentation Task**: Typhoon rainband remote sensing image dataset.

### Hyperparameters
- Optimizer: Adam
- Learning Rate: 1e-3
- Batch Size: 16
- Number of Diffusion Steps (`T`): 2 (Baseline), 5 (Ablation Group 7)
- Training Epochs: 100

### Evaluation Metrics
- **Time-Series Prediction Task**: MSE, RMSE
- **Image Segmentation Task**: Dice coefficient, IoU

---

## Running the Code

```bash
python mian.py 