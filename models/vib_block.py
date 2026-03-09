import torch
import torch.nn as nn

class TensorVIB(nn.Module):
    def __init__(self, channels, clamp_logvar=(-6.0, 0.5)):
        super().__init__()
        self.mu_head = nn.Conv2d(channels, channels, kernel_size=1)
        self.lv_head = nn.Conv2d(channels, channels, kernel_size=1)
        self.clamp_lo, self.clamp_hi = clamp_logvar

    def forward(self, x, sample=True):
        mu = self.mu_head(x)
        logvar = self.lv_head(x)
        logvar = torch.clamp(logvar, self.clamp_lo, self.clamp_hi)
        if sample:
            eps = torch.randn_like(mu)
            t = mu + torch.exp(0.5 * logvar) * eps
        else:
            t = mu
        return t, mu, logvar

def kl_gaussian_standard(mu, logvar, reduction="mean"):
    # 0.5 * (mu^2 + exp(logvar) - logvar - 1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl