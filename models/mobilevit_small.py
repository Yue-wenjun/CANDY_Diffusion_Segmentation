import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MobileViTSmall(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=False):
        """
        保留 bilinear 参数以兼容原 UNet 的初始化调用。
        """
        super(MobileViTSmall, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        # 1. 骨干网络：提取特征，去掉最终的分类池化头
        self.backbone = timm.create_model(
            'mobilevit_s', 
            pretrained=False, 
            in_chans=in_channel, 
            num_classes=0, 
            global_pool=''
        )
        
        # 2. 轻量化解码器：MobileViT-S 最后一层输出通道为 640
        self.decoder = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channel, kernel_size=1)
        )

    def forward(self, x):
        # 提取空间收缩后的全局特征
        features = self.backbone(x) 
        
        # 映射通道数到目标输出通道
        out = self.decoder(features)
        
        # 双线性上采样恢复到输入特征的尺寸 (H, W)
        out = F.interpolate(
            out, 
            size=x.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        return out