import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=False):
        """
        保留 bilinear 参数以兼容原 UNet 的初始化调用。
        """
        super(SegFormerB0, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        # MiT-B0 的标准极简配置 (总参数量约 3.8M)
        config = SegformerConfig(
            num_channels=in_channel,       
            num_labels=out_channel,        
            depths=[2, 2, 2, 2],           
            hidden_sizes=[32, 64, 160, 256],
            decoder_hidden_size=256,       
        )
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        # 1. 前向传播提取特征并过 All-MLP 解码器
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # 输出形状: (B, out_channel, H/4, W/4)
        
        # 2. 双线性上采样恢复到输入特征的尺寸 (H, W)
        logits = F.interpolate(
            logits, 
            size=x.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        return logits