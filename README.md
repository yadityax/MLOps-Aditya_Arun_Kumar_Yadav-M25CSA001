# Assignment 5

This repository contains Assignment 5 implementation:

1. Q1: Fine-tune a pre-trained ViT-S on CIFAR-100 with and without LoRA (PEFT), run LoRA sweeps, and select best settings with Optuna.
2. Q2: Perform adversarial analysis on CIFAR-10 using FGSM (scratch and IBM ART), then train adversarial detectors (PGD and BIM based).


## Setup

### Docker (recommended)

```bash
docker build -t assignment5 .
docker run --rm -it --gpus all -v "$PWD":/workspace assignment5
```

If GPU is not available, remove `--gpus all`.

### Local environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

## Train and Test Commands

### Q1

```bash
# Baseline (without LoRA)
PYTHONPATH=src python -m assignment5.train_q1 --mode baseline --num-workers 0


# Full LoRA sweep (ranks 2,4,8 and alphas 2,4,8)
PYTHONPATH=src python -m assignment5.train_q1 --mode sweep --num-workers 0

# Optuna search
PYTHONPATH=src python -m assignment5.train_q1 --mode optuna --trials 10 --num-workers 0
```

### Q2

```bash
# Q2.1 FGSM (scratch vs ART)
PYTHONPATH=src python -m assignment5.train_q2 --task fgsm --num-workers 0

# Q2.2 Detector with PGD
PYTHONPATH=src python -m assignment5.train_q2 --task detect --attack-type pgd --num-workers 0

# Q2.2 Detector with BIM
PYTHONPATH=src python -m assignment5.train_q2 --task detect --attack-type bim --num-workers 0
```

## Q1 Results

### Baseline (without LoRA)

Only classifier head was trained.

- Test accuracy: `0.8099`

### LoRA Sweep

- Completed combinations: `9/9`
- Sweep grid: `rank={2,4,8}`, `alpha={2,4,8}`, `dropout=0.1`
- Best sweep config: `rank=8`, `alpha=4`, `dropout=0.1`
- Best test accuracy: `0.8984`

### Optuna

- Best trial parameters: `rank=8`, `alpha=4`, `dropout=0.1`
- Best value: `0.8984`

### Key observation

- Baseline accuracy: `0.8099`
- Best LoRA accuracy: `0.8984`
- Absolute gain: `+0.0885`

## Q1 Required Train-Val Table

Experiment No. (based on combinations in step 2): `8`, Rank: `8`, Alpha: `4`

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.976758 | 0.407636 | 0.7704 | 0.8780 |
| 2 | 0.351040 | 0.360579 | 0.8913 | 0.8882 |
| 3 | 0.289491 | 0.342992 | 0.9082 | 0.8920 |
| 4 | 0.245494 | 0.346833 | 0.9206 | 0.8902 |
| 5 | 0.219256 | 0.317761 | 0.9298 | 0.9020 |
| 6 | 0.191144 | 0.340323 | 0.9385 | 0.8962 |
| 7 | 0.170072 | 0.342029 | 0.9450 | 0.8938 |
| 8 | 0.154088 | 0.353369 | 0.9502 | 0.8936 |
| 9 | 0.138923 | 0.338095 | 0.9549 | 0.8976 |
| 10 | 0.126042 | 0.344799 | 0.9600 | 0.8952 |

## Q1 Testing Table (Required Format)

| LoRA layers (with/without) | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Parameters used |
|---|---:|---:|---:|---:|---:|
| Without LoRA | - | - | - | 0.8099 | 38500 |
| With LoRA (Q,K,V) | 2 | 2 | 0.1 | 0.8958 | 93796 |
| With LoRA (Q,K,V) | 2 | 4 | 0.1 | 0.8939 | 93796 |
| With LoRA (Q,K,V) | 2 | 8 | 0.1 | 0.8911 | 93796 |
| With LoRA (Q,K,V) | 4 | 2 | 0.1 | 0.8952 | 149092 |
| With LoRA (Q,K,V) | 4 | 4 | 0.1 | 0.8972 | 149092 |
| With LoRA (Q,K,V) | 4 | 8 | 0.1 | 0.8973 | 149092 |
| With LoRA (Q,K,V) | 8 | 2 | 0.1 | 0.8983 | 259684 |
| With LoRA (Q,K,V) | 8 | 4 | 0.1 | 0.8984 | 259684 |
| With LoRA (Q,K,V) | 8 | 8 | 0.1 | 0.8968 | 259684 |

## Q2 Results

### FGSM: scratch vs IBM ART

- Clean accuracy: `0.8405`
- FGSM (scratch) accuracy: `0.2393`
- FGSM (ART) accuracy: `0.2383`

### Adversarial detector (ResNet34)

- BIM detector reported test accuracy: `0.9994`
- PGD detector final validation accuracy (history): `1.0000`
- BIM detector final validation accuracy (history): `1.0000`
- Requirement `>= 70%` detection accuracy is satisfied.

## Tables, Graphs, and Qualitative Results

### Q1 Tables (CSV)

- [outputs/q1_train_val_table_required_all_experiments.csv](outputs/q1_train_val_table_required_all_experiments.csv)
- [outputs/q1_testing_table_required.csv](outputs/q1_testing_table_required.csv)
- [outputs/q1_lora_sweep_summary.csv](outputs/q1_lora_sweep_summary.csv)

### Q1 Graphs

![Q1 Training Curves](outputs/q1_lora_r8_a4_d0.1/training_curves.png)
![Q1 Classwise Accuracy](outputs/q1_lora_r8_a4_d0.1/classwise_test_accuracy.png)
![Q1 LoRA Gradient Norms](outputs/q1_lora_r8_a4_d0.1/lora_gradient_norms.png)

### Q2 Tables and Graphs

- [outputs/q2_fgsm/attack_results.csv](outputs/q2_fgsm/attack_results.csv)
- [outputs/q2_detector_pgd/pgd_detector_history.csv](outputs/q2_detector_pgd/pgd_detector_history.csv)
- [outputs/q2_detector_bim/bim_detector_history.csv](outputs/q2_detector_bim/bim_detector_history.csv)

![Q2 FGSM Comparison](outputs/q2_fgsm/fgsm_comparison.png)
![Q2 Adversarial Samples](outputs/q2_fgsm/adversarial_samples.png)

## Model Weights

### Q1 best model

- [outputs/q1_lora_r8_a4_d0.1/best_model.pth](outputs/q1_lora_r8_a4_d0.1/best_model.pth)

### Q2 all models

- [outputs/q2_fgsm/best_clean_model.pth](outputs/q2_fgsm/best_clean_model.pth)
- [outputs/q2_detector_pgd/best_pgd_detector.pth](outputs/q2_detector_pgd/best_pgd_detector.pth)
- [outputs/q2_detector_bim/best_bim_detector.pth](outputs/q2_detector_bim/best_bim_detector.pth)

## External Links

- WandB project/run links: `[https://wandb.ai/m25csa001-indian-institute-of-technology-jodhpur/assignment5?nw=nwuserm25csa001]`
- HuggingFace model link (best Q1): `[https://huggingface.co/m25csa001/assignment5-weights]`



