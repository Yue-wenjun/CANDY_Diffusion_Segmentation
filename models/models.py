import torch
import torch.nn as nn
import torch.nn.functional as F
from models.diffusion import DiffusionModel

class DiffusionModelWrapper:
    def __init__(self, base_config):
        self.base_config = base_config
        
    def create_model(self, ablation_type, **kwargs):
        """
        根据消融类型和 config.py 传入的 override_config 创建模型变体
        """
        # 1. 基础配置合并 override 配置 (例如更换 decoder_type, 修改 T 等)
        config = self.base_config.copy()
        config.update(kwargs)
        
        # ==========================================
        # 通用结构模型 (仅参数/模块替换，网络拓扑流向不变)
        # ==========================================
        standard_models = ["baseline", "segformer", "mobilevit", "adjust_steps"]
        if ablation_type in standard_models:
            return DiffusionModel(**config)
            
        # ==========================================
        # 拓扑结构消融实验 (重写了 forward 逻辑的特殊变体)
        # ==========================================
        elif ablation_type == "no_skip":
            class NoSkipDiffusionModel(DiffusionModel):
                def forward(self, x, graph_schedule=None):
                    device = x.device
                    origin = torch.zeros(
                        self.T, self.batch_size, self.in_channel, self.hidden_size, self.input_size
                    ).to(device)
                    input = x

                    for t in range(self.T):
                        output = self.candies[t](input)
                        origin[t] = input
                        input = output

                    if graph_schedule is None:
                        graph_schedule = torch.linspace(0.7, 0.2, self.T).to(device)

                    # 反向过程移除了跳跃连接与特征融合
                    for t in reversed(range(self.T)):
                        reverse_input = input  # 直接将上一步输出作为输入
                        output = self.unets[t](reverse_input)
                        input = output

                    output_seg = self.seg_head(output)
                    return torch.sum(output_seg, dim=1, keepdim=True)
            
            return NoSkipDiffusionModel(**config)
            
        elif ablation_type == "simple_cnn":
            class SimpleCNNDiffusionModel(DiffusionModel):
                def __init__(self, **model_cfg):
                    super().__init__(**model_cfg)
                    # 替换 CANDY 模块为简单的双层 CNN
                    self.candies = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(model_cfg.get("in_channel", 1), model_cfg.get("hidden_channel", 64), 
                                     kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(model_cfg.get("hidden_channel", 64), model_cfg.get("out_channel", 1), 
                                     kernel_size=3, stride=1, padding=1)
                        )
                        for _ in range(model_cfg["T"])
                    ])
            
            return SimpleCNNDiffusionModel(**config)
            
        elif ablation_type == "simple_decoder":
            class SimpleDecoderDiffusionModel(DiffusionModel):
                def __init__(self, **model_cfg):
                    # 强行指定一个内部状态以绕过检查，下面立即覆盖 unets
                    model_cfg["decoder_type"] = "simple_decoder" 
                    super().__init__(**model_cfg)
                    
                    # 替换复杂 Decoder 为单层 1x1 卷积
                    self.unets = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(model_cfg["in_channel"], model_cfg["out_channel"], kernel_size=1),
                            nn.ReLU()
                        )
                        for _ in range(model_cfg["T"])
                    ])
            
            return SimpleDecoderDiffusionModel(**config)
            
        elif ablation_type == "sde":
            class SDEDiffusionModel(DiffusionModel):
                def forward(self, x, graph_schedule=None):
                    device = x.device
                    origin = torch.zeros(
                        self.T, self.batch_size, self.in_channel, self.hidden_size, self.input_size
                    ).to(device)
                    input = x

                    # 前向扩散：注入随机噪声 (SDE 特性)
                    for t in range(self.T):
                        output = self.candies[t](input)
                        noise = torch.randn_like(output) * 0.1  # 注入噪声
                        output = output + noise
                        origin[t] = input
                        input = output

                    if graph_schedule is None:
                        graph_schedule = torch.linspace(0.7, 0.2, self.T).to(device)

                    for t in reversed(range(self.T)):
                        graph_factor = graph_schedule[t]
                        # SDE 实验中使用了简化版的反向加权逻辑
                        reverse_input = (1 - graph_factor) * input + graph_factor * origin[t]
                        output = self.unets[t](reverse_input)
                        input = output

                    output_seg = self.seg_head(output)
                    return torch.sum(output_seg, dim=1, keepdim=True)
            
            return SDEDiffusionModel(**config)
            
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")