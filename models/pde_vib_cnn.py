# models/pde_vib_cnn.py
import torch
import torch.nn as nn
from models.pde_trainable import TrainablePDEBank      # or MixturePDEBank if you switched
from models.vib_block import TensorVIB, kl_gaussian_standard
from models.nets import make_resnet18_headless

class ModelPDE_VIB_CNN(nn.Module):
    def __init__(self, pde_out_ch=32, pde_steps=1, num_classes=10,
                 kl_reduce="mean", pde_init_lambda=0.01, pde_use_bn=True,
                 use_conv_stem=True,  # <— enable the stem
                 bottleneck_ratio=0.5, # <— C -> C/2 -> C
                 residual_alpha_init=0.95):
        super().__init__()
        self.pde = TrainablePDEBank(in_ch=3, out_ch=pde_out_ch, repeat=pde_steps,
                                    init_lambda=pde_init_lambda, use_bn=pde_use_bn)

        # (A) small learned adaptation before VIB (accuracy-friendly)
        self.conv_stem = nn.Identity()
        if use_conv_stem:
            self.conv_stem = nn.Sequential(
                nn.Conv2d(pde_out_ch, pde_out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(pde_out_ch),
                nn.ReLU(inplace=True)
            )

        # (B) channel bottleneck: C -> Cb -> VIB -> C
        C  = pde_out_ch
        Cb = max(4, int(C * bottleneck_ratio))  # avoid too small; floor at 4
        self.pre_bn  = nn.Conv2d(C,  Cb, kernel_size=1, bias=False)
        self.post_bn = nn.Conv2d(Cb, C,  kernel_size=1, bias=False)

        # VIB operates on the **compressed channels**
        self.vib = TensorVIB(channels=Cb)   # same TensorVIB as before

        # residual gate (helps preserve accuracy)
        self.alpha = nn.Parameter(torch.tensor(residual_alpha_init))

        self.backbone = make_resnet18_headless(in_ch=pde_out_ch, num_classes=num_classes)
        self.kl_reduce = kl_reduce

    def forward(self, x, sample=True):
        f = self.pde(x)               # (N, C, H, W)
        f = self.conv_stem(f)         # (N, C, H, W)
        z = self.pre_bn(f)            # (N, Cb, H, W)
        t, mu, logvar = self.vib(z, sample=sample)  # (N, Cb, H, W)
        t = self.post_bn(t)           # (N, C, H, W)
        # residual blend: pass some clean path if VIB compresses too hard
        t = self.alpha * t + (1.0 - self.alpha) * f
        logits = self.backbone(t)
        return logits, mu, logvar     # KL computed on (mu, logvar) in Cb-space

    def kl(self, mu, logvar):
        return kl_gaussian_standard(mu, logvar, reduction=self.kl_reduce)
