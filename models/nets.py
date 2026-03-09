import torch.nn as nn
from torchvision.models import resnet18

def make_resnet18_headless(in_ch=32, num_classes=10):
    m = resnet18(num_classes=num_classes)
    # CIFAR-friendly stem
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m
