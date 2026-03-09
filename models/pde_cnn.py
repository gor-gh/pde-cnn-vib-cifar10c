import torch.nn as nn
from models.pde_trainable import TrainablePDEBank
from models.nets import make_resnet18_headless

class ModelPDE_CNN(nn.Module):
    def __init__(self, pde_out_ch=32, pde_steps=1, num_classes=10,
                 pde_init_lambda=0.07, pde_use_bn=True, use_conv_stem=False):
        super().__init__()
        self.pde = TrainablePDEBank(in_ch=3, out_ch=pde_out_ch, repeat=pde_steps,
                                    init_lambda=pde_init_lambda, use_bn=pde_use_bn)

        self.stem = nn.Identity()
        if use_conv_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(pde_out_ch, pde_out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(pde_out_ch),
                nn.ReLU(inplace=True)
            )

        self.backbone = make_resnet18_headless(in_ch=pde_out_ch, num_classes=num_classes)

    def forward(self, x):
        f = self.pde(x)
        f = self.stem(f)
        return self.backbone(f)
