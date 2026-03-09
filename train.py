import time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.nets import make_resnet18_headless
from tqdm import tqdm
import math
from utils import expected_calibration_error, average_nll, ensure_dir, csv_writer

LABEL_SMOOTH = 0.05

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def loaders(batch_size=128):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
    ])
    train = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm_train)
    test  = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm_test)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True), \
           DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

def step_beta(epoch, total_warm=15, beta_max=3e-4):
    # linear warmup; clamp afterwards
    if epoch < total_warm:
        return beta_max * (epoch+1) / total_warm
    return beta_max

@torch.no_grad()
def eval_top1(model, loader, device, is_vib=False):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_vib:
            logits, _, _ = model(x, sample=False)  # deterministic t = mu
        else:
            logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total

def train_variant(variant="baseline", epochs=20, pde_steps=3, pde_out_ch=32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, test_loader = loaders()

    if variant == "baseline":
        # CIFAR-friendly ResNet18 stem (no 7x7 + maxpool) for fair comparison
        model = make_resnet18_headless(in_ch=3, num_classes=10)
        params = count_parameters(model)
        print(f"Trainable parameters: {params}")
        is_vib = False
    elif variant == "pde_cnn":
        from models.pde_cnn import ModelPDE_CNN
        model = ModelPDE_CNN(pde_out_ch=pde_out_ch, pde_steps=pde_steps, num_classes=10)
        params = count_parameters(model)
        print(f"Trainable parameters: {params}")
        is_vib = False
    elif variant == "pde_vib_cnn":
        from models.pde_vib_cnn import ModelPDE_VIB_CNN
        model = ModelPDE_VIB_CNN(pde_out_ch=pde_out_ch, pde_steps=pde_steps, num_classes=10)
        params = count_parameters(model)
        print(f"Trainable parameters: {params}")
        is_vib = True
    else:
        raise ValueError("unknown variant")

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # CSV logging
    ensure_dir("results")
    path = f"results/metrics_{variant}.csv"
    header = ["epoch", "split", "acc", "nll", "ece", "gap"]
    fh, writer = csv_writer(path, header)

    total_train_start = time.time()
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        beta = step_beta(epoch) if is_vib else 0.0
        pbar = tqdm(train_loader, desc=f"{variant} epoch {epoch+1}/{epochs} β={beta:.4g}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if is_vib:
                logits, mu, logvar = model(x, sample=True)
                ce = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)
                kl = model.kl(mu, logvar)  # KL reduction = 'mean' by default
                loss = ce + beta * kl
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)
            loss.backward()
            opt.step()

        # ---- end epoch: evaluate & log ----
        train_acc, train_nll, train_ece = eval_metrics(model, train_loader, device, is_vib=is_vib)
        test_acc,  test_nll,  test_ece  = eval_metrics(model,  test_loader, device, is_vib=is_vib)
        gap = train_acc - test_acc

        # write two rows (train/test) so plotting is trivial later
        writer.writerow([epoch+1, "train", f"{train_acc:.4f}", f"{train_nll:.6f}", f"{train_ece:.6f}", f"{gap:.4f}"])
        writer.writerow([epoch+1, "test",  f"{test_acc:.4f}",  f"{test_nll:.6f}",  f"{test_ece:.6f}",  f"{gap:.4f}"])
        fh.flush()

        print(f"[{variant}] epoch {epoch+1}: "
              f"train_acc={train_acc:.2f} test_acc={test_acc:.2f} gap={gap:.2f} "
              f"test_nll={test_nll:.4f} test_ece={test_ece:.4f}")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"[{variant}] epoch {epoch + 1} time: {epoch_time:.2f}s")


    # Save final checkpoint for later CIFAR-10-C evaluation
    ensure_dir("checkpoints")
    ckpt_path = f"checkpoints/{variant}_s2_b3e4_final.pt"
    torch.save({
        "variant": variant,
        "epoch": epochs,
        "model": model.state_dict(),
        "pde_steps": pde_steps,
        "pde_out_ch": pde_out_ch,
        "is_vib": is_vib,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    total_train_time = time.time() - total_train_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    inf_total, inf_per_batch, inf_per_image = measure_inference_time(
        model, test_loader, device, is_vib=is_vib
    )

    print(f"[{variant}] total training time: {total_train_time:.2f}s")
    print(f"[{variant}] average epoch time: {avg_epoch_time:.2f}s")
    print(f"[{variant}] total inference time (test set): {inf_total:.4f}s")
    print(f"[{variant}] inference time per batch: {inf_per_batch:.6f}s")
    print(f"[{variant}] inference time per image: {inf_per_image * 1000:.6f} ms")

    timing_path = f"results/timing_{variant}.csv"
    tfh, twriter = csv_writer(
        timing_path,
        [
            "variant",
            "params",
            "total_train_time_sec",
            "avg_epoch_time_sec",
            "total_inference_time_sec",
            "inference_time_per_batch_sec",
            "inference_time_per_image_ms",
        ],
    )
    twriter.writerow([
        variant,
        params,
        f"{total_train_time:.6f}",
        f"{avg_epoch_time:.6f}",
        f"{inf_total:.6f}",
        f"{inf_per_batch:.6f}",
        f"{inf_per_image * 1000:.6f}",
    ])
    tfh.close()

@torch.no_grad()
def eval_metrics(model, loader, device, is_vib: bool):
    """
    Returns: acc (float %), nll (float), ece (float)
    Uses deterministic VIB at eval (t = mu).
    """
    model.eval()
    correct, total = 0, 0
    all_logits, all_labels = [], []

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

@torch.no_grad()
def measure_inference_time(model, loader, device, is_vib: bool):
    """
    Returns:
        total_time_sec,
        time_per_batch_sec,
        time_per_image_sec
    """
    model.eval()
    total_images = 0

    # Optional warm-up for more stable timing on MPS/CUDA
    warmup_done = False

    start = time.time()
    for x, y in loader:
        x = x.to(device)

        if not warmup_done:
            if is_vib:
                _ = model(x, sample=False)
            else:
                _ = model(x)
            warmup_done = True

        if is_vib:
            _ = model(x, sample=False)
        else:
            _ = model(x)

        total_images += x.size(0)
    total_time = time.time() - start

    num_batches = len(loader)
    time_per_batch = total_time / num_batches
    time_per_image = total_time / total_images

    return total_time, time_per_batch, time_per_image

if __name__ == "__main__":
    EPOCHS = 15

    # 1) ResNet baseline
    # train_variant("baseline", epochs=EPOCHS)

    # 2) PDE + CNN
    # train_variant("pde_cnn", epochs=EPOCHS, pde_steps=1, pde_out_ch=32)

    # 3) PDE + VIB + CNN
    train_variant("pde_vib_cnn", epochs=EPOCHS, pde_steps=2, pde_out_ch=32)