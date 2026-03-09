# PDE-CNN-VIB: Robust Image Classification under Common Corruptions

This repository contains the official implementation for the paper:

**Improving Robust Image Classification under Common Corruptions:  
A PDE-Regularized Variational Information Bottleneck Network**

The proposed architecture combines **Partial Differential Equation (PDE) based regularization** with a **Variational Information Bottleneck (VIB)** to improve the robustness of convolutional neural networks against common image corruptions.

The model is evaluated on **CIFAR-10** and the **CIFAR-10-C** corruption benchmark.

---

## Architecture

The proposed pipeline follows the structure:

Input → PDE Regularization → CNN Backbone → VIB Compression → Classifier

The PDE module introduces diffusion-based smoothing while the VIB module encourages compression of irrelevant features.

---

## Repository Structure

## Repository Structure

```text
CSIT-code-cifar10c-ready/
├── datasets/
│   └── cifar10c.py
├── models/
│   ├── nets.py
│   ├── pde_layers.py
│   ├── pde_trainable.py
│   ├── pde_cnn.py
│   ├── pde_vib_cnn.py
│   └── vib_block.py
├── train.py
├── eval_cifar10c.py
├── utils.py
├── config.yaml
└── requirements.txt
```

---

## Installation

Create a Python environment and install the required dependencies.

Recommended Python version:

Python 3.11+

Install dependencies using:

```bash
pip install -r requirements.txt
```
---

## Dataset

The experiments use the following datasets:

- **CIFAR-10**
- **CIFAR-10-C**

CIFAR-10 will automatically be downloaded when running the training script.

The CIFAR-10-C corruption benchmark can be obtained from:

https://github.com/hendrycks/robustness

After downloading, place the corruption files in the directory expected by the script `datasets/cifar10c.py`.

---

## Training

To train the baseline CNN model, run:

```bash
python train.py
```
Inside `train.py`, different model variants can be selected by modifying the `train_variant` call:

```text
train_variant("baseline")
train_variant("pde_cnn")
train_variant("pde_vib_cnn")
```

The script will automatically download CIFAR-10, train the selected model, and save the checkpoint to the `checkpoints/` directory.

---

## Robustness Evaluation

To evaluate robustness on the CIFAR-10-C benchmark, run:

```bash
python eval_cifar10c.py
```

This script evaluates the trained model across the corruption types included in CIFAR-10-C and reports the classification accuracy for each corruption as well as the `mean corruption accuracy (mCA)` metric.

The evaluation is performed at corruption `severity level 3`, which provides a balanced assessment of model robustness under moderate image degradation.

---

## Results

The proposed PDE-CNN-VIB architecture improves robustness to common image corruptions while introducing only a small computational overhead.

| Model | Clean Accuracy (%) | mCA (%) |
|------|--------------------|--------|
| Baseline CNN | 80.60 | 65.86 |
| PDE-CNN-VIB | 88.52 | 68.88 |

In addition to improved classification performance, the proposed architecture demonstrates stronger robustness under noise-related corruptions in the CIFAR-10-C benchmark.
---

## License

This project is released under the MIT License.