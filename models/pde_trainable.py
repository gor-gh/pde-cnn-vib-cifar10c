# models/pde_trainable.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def laplacian_kernel():
    k = torch.tensor([[0., 1., 0.],
                      [1.,-4., 1.],
                      [0., 1., 0.]], dtype=torch.float32)
    return k

class TrainablePDEBank(nn.Module):
    """
    Expand RGB->C via 1x1 conv (trainable), then apply a PDE step with per-channel
    learnable λ (>=0 via softplus). Repeat a few steps (repeat=1..3).
    Optional BN+ReLU help optimization without breaking PDE flavor.
    """
    def __init__(self, in_ch=3, out_ch=32, repeat=1, init_lambda=0.01, use_bn=True):
        super().__init__()
        self.out_ch, self.repeat, self.use_bn = out_ch, repeat, use_bn

        # Trainable expand to widen channels (cheap)
        self.expand = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

        with torch.no_grad():
            self.expand.weight.zero_()
            # replicate RGB into out_ch evenly
            for oc in range(self.out_ch):
                self.expand.weight[oc, oc % 3, 0, 0] = 1.0
            if self.expand.bias is not None:
                self.expand.bias.zero_()

        # Fixed depthwise Laplacian kernel, replicated per channel
        k = laplacian_kernel()[None, None, :, :]              # (1,1,3,3)
        self.register_buffer("k_pde", k.repeat(out_ch, 1, 1, 1))  # (C,1,3,3)

        # Learnable per-channel λ (softplus keeps it >= 0)
        init_raw = torch.log(torch.exp(torch.tensor(init_lambda)) - 1.0)  # inverse softplus
        self.l_raw = nn.Parameter(init_raw.expand(out_ch).clone())        # (C,)

        # Optional normalization + nonlinearity around PDE step
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def _lambda(self):
        return F.softplus(self.l_raw).view(1, -1, 1, 1)  # (1,C,1,1)

    def forward_one(self, x):
        lap = F.conv2d(x, self.k_pde, stride=1, padding=1, groups=self.out_ch)  # depthwise
        y = x + self._lambda() * lap
        if self.use_bn:
            y = self.bn(y)
        return self.act(y)

    def forward(self, x):
        x = self.expand(x)
        for _ in range(self.repeat):
            x = self.forward_one(x)
        return x
