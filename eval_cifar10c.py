
import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from datasets.cifar10c import CIFAR10C
from utils import expected_calibration_error, average_nll

from models.pde_cnn import ModelPDE_CNN
from models.pde_vib_cnn import ModelPDE_VIB_CNN
from models.nets import make_resnet18_headless

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise",
    "defocus_blur","glass_blur","motion_blur","zoom_blur",
    "snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression",
]

@torch.no_grad()
def eval_metrics(model, loader, device, is_vib: bool):
    model.eval()
    all_logits, all_labels = [], []
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_vib:
            logits, _, _ = model(x, sample=False)
        else:
            logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_logits.append(logits)
        all_labels.append(y)

    acc = 100.0 * correct / total
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    nll = average_nll(logits_cat, labels_cat)
    ece = expected_calibration_error(logits_cat, labels_cat, n_bins=15)
    return acc, nll, ece

def build_model(variant: str, pde_steps: int, pde_out_ch: int):
    if variant == "baseline":
        model = make_resnet18_headless(in_ch=3, num_classes=10)
        return model, False
    if variant == "pde_cnn":
        return ModelPDE_CNN(pde_out_ch=pde_out_ch, pde_steps=pde_steps, num_classes=10), False
    if variant == "pde_vib_cnn":
        return ModelPDE_VIB_CNN(pde_out_ch=pde_out_ch, pde_steps=pde_steps, num_classes=10), True
    raise ValueError("variant must be one of: baseline, pde_cnn, pde_vib_cnn")

def get_device():
    # Keep your original preference: MPS then CPU, but allow CUDA if present (harmless).
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pde_vib_cnn", choices=["baseline","pde_cnn","pde_vib_cnn"])
    ap.add_argument("--ckpt", required=True, help="Path to a .pt checkpoint (state_dict or dict with 'model').")
    ap.add_argument("--data_root", default="data", help="Folder containing CIFAR-10 and CIFAR-10-C.")
    ap.add_argument("--severity", type=int, default=3, choices=[1,2,3,4,5])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pde_steps", type=int, default=1)
    ap.add_argument("--pde_out_ch", type=int, default=32)
    ap.add_argument("--corruptions", default="all", help="Comma-separated list or 'all'.")
    args = ap.parse_args()

    device = get_device()

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # Clean CIFAR-10 test
    clean_test = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=tfm)
    clean_loader = DataLoader(clean_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model, is_vib = build_model(args.variant, args.pde_steps, args.pde_out_ch)
    model.to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)

    clean_acc, clean_nll, clean_ece = eval_metrics(model, clean_loader, device, is_vib=is_vib)
    print(f"CLEAN: acc={clean_acc:.2f}  nll={clean_nll:.4f}  ece={clean_ece:.4f}")

    if args.corruptions == "all":
        corrs = CORRUPTIONS
    else:
        corrs = [c.strip() for c in args.corruptions.split(",") if c.strip()]

    rows = []
    accs = []
    for c in corrs:
        dset = CIFAR10C(root=args.data_root, corruption=c, severity=args.severity, transform=tfm)
        loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        acc, nll, ece = eval_metrics(model, loader, device, is_vib=is_vib)
        accs.append(acc)
        rows.append((c, args.severity, acc, nll, ece))
        print(f"{c:>18} (s={args.severity}): acc={acc:.2f}  nll={nll:.4f}  ece={ece:.4f}")

    if rows:
        mca = sum(accs)/len(accs)
        print(f"mCA over {len(rows)} corruptions @ severity {args.severity}: {mca:.2f}")

if __name__ == "__main__":
    main()
