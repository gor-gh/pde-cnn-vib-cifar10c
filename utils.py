import os, csv
import torch
import torch.nn.functional as F

def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Multiclass ECE with max-prob binning.
    logits: [N, K], labels: [N]
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)           # [N, K]
        confidences, preds = probs.max(dim=1)          # [N], [N]
        labels = labels.view(-1)
        N = labels.numel()

        ece = torch.tensor(0.0, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

        for b in range(n_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b+1]
            in_bin = (confidences > lo) & (confidences <= hi)
            bin_count = in_bin.sum().item()
            if bin_count == 0:
                continue
            acc_bin = (preds[in_bin] == labels[in_bin]).float().mean()
            conf_bin = confidences[in_bin].mean()
            ece += (bin_count / N) * (acc_bin - conf_bin).abs()
        return ece.item()

def average_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Mean Negative Log-Likelihood = mean cross-entropy on the split.
    """
    with torch.no_grad():
        # sum over batch then divide by N for stable average
        nll_sum = F.cross_entropy(logits, labels, reduction="sum")
        return (nll_sum / labels.numel()).item()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def csv_writer(path: str, header: list[str]):
    """
    Open a CSV file (create if missing) and return (file_handle, writer).
    If file exists but empty, write header.
    """
    file_exists = os.path.exists(path)
    fh = open(path, "a", newline="")
    writer = csv.writer(fh)
    if (not file_exists) or (os.path.getsize(path) == 0):
        writer.writerow(header)
    return fh, writer
