import torch
import torch.nn as nn
from models.diffusion import DiffusionModel

# Keys that DiffusionModel.__init__ accepts; everything else is filtered out.
_DIFFUSION_KEYS = {
    "batch_size", "in_channel", "hidden_channel", "out_channel",
    "input_size", "hidden_size", "T", "num_classes", "decoder_type",
}


def _diffusion_cfg(config: dict) -> dict:
    return {k: v for k, v in config.items() if k in _DIFFUSION_KEYS}


class DiffusionModelWrapper:
    def __init__(self, base_config):
        self.base_config = base_config

    def create_model(self, ablation_type):
        cfg = _diffusion_cfg(self.base_config)

        # ── Standard variants: decoder swapped via decoder_type in config ──────
        # "segformer"  → cfg["decoder_type"] == "segformer_b0"   (set in config.py)
        # "mobilevit"  → cfg["decoder_type"] == "mobilevit_small"
        # "baseline" / "adjust_steps" → decoder_type == "unet"
        if ablation_type in ("baseline", "segformer", "mobilevit", "adjust_steps"):
            return DiffusionModel(**cfg)

        # ── Ablation: no skip connections ─────────────────────────────────────
        elif ablation_type == "no_skip":
            class NoSkipDiffusionModel(DiffusionModel):
                def forward(self, x, graph_schedule=None):
                    device = x.device
                    B = x.shape[0]
                    feat = x
                    t_idx = torch.zeros(B, dtype=torch.long, device=device)
                    for t in range(self.T):
                        feat = self.candies[t](feat)   # forward, no origin saved
                    if graph_schedule is None:
                        graph_schedule = torch.linspace(0.7, 0.2, self.T, device=device)
                    for t in reversed(range(self.T)):
                        t_idx.fill_(t)
                        feat = self.unet(feat, t_idx)  # no fusion, no skip
                    return self.seg_head(feat)

            return NoSkipDiffusionModel(**cfg)

        # ── Ablation: replace CANDY with simple double-conv ───────────────────
        elif ablation_type == "simple_cnn":
            class SimpleCNNDiffusionModel(DiffusionModel):
                def __init__(self, **c):
                    super().__init__(**c)
                    hc = c.get("hidden_channel", 16)
                    self.candies = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(c["in_channel"], hc, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hc, c["out_channel"], 3, padding=1),
                        )
                        for _ in range(c["T"])
                    ])

            return SimpleCNNDiffusionModel(**cfg)

        # ── Ablation: replace shared UNet with a single 1×1 conv ─────────────
        elif ablation_type == "simple_decoder":
            class SimpleDecoderDiffusionModel(DiffusionModel):
                def __init__(self, **c):
                    super().__init__(**c)
                    self.unet = nn.Sequential(
                        nn.Conv2d(c["in_channel"], c["out_channel"], kernel_size=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x, graph_schedule=None):
                    device = x.device
                    B = x.shape[0]
                    origin = torch.zeros(
                        self.T, B, self.in_channel, self.hidden_size, self.input_size, device=device
                    )
                    feat = x
                    for t in range(self.T):
                        origin[t] = feat
                        feat = self.candies[t](feat)
                    if graph_schedule is None:
                        graph_schedule = torch.linspace(0.7, 0.2, self.T, device=device)
                    for t in reversed(range(self.T)):
                        fused = self.fusion_convs[t](torch.cat([feat, origin[t]], dim=1))
                        reverse_input = (1 - graph_schedule[t]) * fused + graph_schedule[t] * origin[t]
                        feat = self.unet(reverse_input)   # no t_idx: 1×1 conv ignores it
                    return self.seg_head(feat)

            return SimpleDecoderDiffusionModel(**cfg)

        # ── Ablation: SDE — inject Gaussian noise in the forward pass ─────────
        elif ablation_type == "sde":
            class SDEDiffusionModel(DiffusionModel):
                def forward(self, x, graph_schedule=None):
                    device = x.device
                    B = x.shape[0]
                    origin = torch.zeros(
                        self.T, B, self.in_channel, self.hidden_size, self.input_size, device=device
                    )
                    feat = x
                    for t in range(self.T):
                        origin[t] = feat
                        feat = self.candies[t](feat) + torch.randn_like(feat) * 0.1
                    if graph_schedule is None:
                        graph_schedule = torch.linspace(0.7, 0.2, self.T, device=device)
                    t_idx = torch.zeros(B, dtype=torch.long, device=device)
                    for t in reversed(range(self.T)):
                        t_idx.fill_(t)
                        fused = self.fusion_convs[t](torch.cat([feat, origin[t]], dim=1))
                        reverse_input = (1 - graph_schedule[t]) * fused + graph_schedule[t] * origin[t]
                        feat = self.unet(reverse_input, t_idx)
                    return self.seg_head(feat)

            return SDEDiffusionModel(**cfg)

        # ── DDPM baseline: Gaussian forward process instead of CANDY ────────────
        # Same shared UNet + fusion_convs + seg_head; only the forward chain changes.
        # This tests whether CANDY's learned deterministic forward is better than
        # standard DDPM Gaussian noise schedule.
        elif ablation_type == "ddpm":
            class DDPMDiffusionModel(DiffusionModel):
                def __init__(self, **c):
                    super().__init__(**c)
                    T = c["T"]
                    # Linear beta schedule (small values → mild noise for feature maps)
                    betas = torch.linspace(1e-4, 2e-2, T)
                    alphas_bar = torch.cumprod(1.0 - betas, dim=0)
                    self.register_buffer("_sqrt_ab",     torch.sqrt(alphas_bar))
                    self.register_buffer("_sqrt_1m_ab",  torch.sqrt(1.0 - alphas_bar))

                def forward(self, x, graph_schedule=None):
                    device = x.device
                    B = x.shape[0]
                    origin = torch.zeros(
                        self.T, B, self.in_channel, self.hidden_size, self.input_size, device=device
                    )
                    feat = x
                    for t in range(self.T):
                        origin[t] = feat
                        # q(x_t | x_0) = sqrt(ā_t)·x_0 + sqrt(1−ā_t)·ε
                        feat = self._sqrt_ab[t] * feat + self._sqrt_1m_ab[t] * torch.randn_like(feat)
                    if graph_schedule is None:
                        graph_schedule = torch.linspace(0.7, 0.2, self.T, device=device)
                    t_idx = torch.zeros(B, dtype=torch.long, device=device)
                    for t in reversed(range(self.T)):
                        t_idx.fill_(t)
                        fused = self.fusion_convs[t](torch.cat([feat, origin[t]], dim=1))
                        reverse_input = (1 - graph_schedule[t]) * fused + graph_schedule[t] * origin[t]
                        feat = self.unet(reverse_input, t_idx)
                    return self.seg_head(feat)

            return DDPMDiffusionModel(**cfg)

        else:
            raise ValueError(
                f"Unknown ablation type: '{ablation_type}'\n"
                f"Available: baseline, segformer, mobilevit, adjust_steps, ddpm, "
                f"no_skip, simple_cnn, simple_decoder, sde"
            )
