from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from assignment5.attacks import (
    art_bim_attack,
    art_fgsm_attack,
    art_pgd_attack,
    fgsm_attack,
    iterative_fgsm_attack,
)
from assignment5.config import AttackConfig, TrainingConfig
from assignment5.data import build_cifar10_loaders
from assignment5.engine import evaluate, save_checkpoint, train_epoch
from assignment5.models import build_resnet18, build_resnet34_detector
from assignment5.utils import default_output_dir, resolve_device, set_seed


def _init_wandb(project: str, entity: str | None, run_name: str | None, tags: list[str] | None):
    try:
        import wandb
    except ImportError:  # pragma: no cover - optional dependency
        return None
    wandb.init(project=project, entity=entity, name=run_name, tags=tags or [])
    return wandb


def _log_wandb_image(wandb, key: str, path: Path) -> None:
    if wandb is None:
        return
    wandb.log({key: wandb.Image(str(path))})


def _plot_sample_grid(
    clean_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    attack_label: str,
    output_dir: Path,
) -> Path:
    sample_count = min(10, clean_images.shape[0], adversarial_images.shape[0])
    figure, axes = plt.subplots(2, sample_count, figsize=(3.2 * sample_count, 6))
    figure.suptitle(f"Clean vs Adversarial Images ({attack_label})")
    denorm = lambda x: (x + 1) / 2

    for index in range(sample_count):
        axes[0, index].imshow(denorm(clean_images[index]).permute(1, 2, 0).clamp(0, 1))
        axes[0, index].set_title("Clean")
        axes[0, index].axis("off")

        axes[1, index].imshow(denorm(adversarial_images[index]).permute(1, 2, 0).clamp(0, 1))
        axes[1, index].set_title(attack_label)
        axes[1, index].axis("off")

    figure.tight_layout()
    path = output_dir / f"{attack_label.lower().replace(' ', '_')}_samples.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


@dataclass(slots=True)
class Q2CleanTrainResult:
    history: pd.DataFrame
    test_metrics: dict[str, float]
    checkpoint_path: Path
    output_dir: Path


@dataclass(slots=True)
class Q2AttackResult:
    clean_accuracy: float
    adversarial_accuracy: float
    perturbation_norms: np.ndarray
    adversarial_images: torch.Tensor | None = None
    clean_images: torch.Tensor | None = None


def _train_resnet18_clean(
    training_config: TrainingConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Q2CleanTrainResult]:
    """Train ResNet18 on clean CIFAR-10 samples."""
    set_seed(training_config.seed)
    data = build_cifar10_loaders(
        training_config.data_root,
        training_config.batch_size,
        training_config.num_workers,
        image_size=224,
    )
    model = build_resnet18(num_classes=data.num_classes, pretrained=False)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    rows: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_checkpoint_path = output_dir / "best_clean_model.pth"

    for epoch in range(1, training_config.epochs + 1):
        train_result = train_epoch(model, data.train_loader, device, optimizer)
        val_metrics = evaluate(model, data.val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_result.metrics.loss,
            "train_accuracy": train_result.metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
        }
        rows.append(row)
        if val_metrics.accuracy >= best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            save_checkpoint(model, output_dir, "best_clean_model.pth")

    history = pd.DataFrame(rows)
    test_metrics = evaluate(model, data.test_loader, device)
    return model, Q2CleanTrainResult(
        history=history,
        test_metrics={"loss": test_metrics.loss, "accuracy": test_metrics.accuracy},
        checkpoint_path=best_checkpoint_path,
        output_dir=output_dir,
    )


def _evaluate_attack(
    model: torch.nn.Module,
    attack_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    loader: Iterable,
    device: torch.device,
    max_samples: int = 1000,
) -> tuple[float, np.ndarray, torch.Tensor, torch.Tensor]:
    """Evaluate attack and measure perturbation norm."""
    model.eval()
    correct = 0
    total = 0
    perturbations = []
    adversarial_batch = None
    clean_batch = None

    for inputs, targets in loader:
        if total >= max_samples:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Attack generation needs gradients.
        with torch.enable_grad():
            adv = attack_fn(model, inputs, targets)
        adv = torch.clamp(adv, 0.0, 1.0)

        # Model evaluation on adversarial inputs does not need gradients.
        with torch.no_grad():
            outputs = model(adv)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            predictions = logits.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += inputs.size(0)

        delta = (adv - inputs).detach().cpu().flatten(start_dim=1)
        perturbations.append(torch.linalg.vector_norm(delta, dim=1).numpy())
        if adversarial_batch is None:
            adversarial_batch = adv.detach().cpu()
            clean_batch = inputs.detach().cpu()

    accuracy = correct / max(total, 1)
    perturbation_array = np.concatenate(perturbations) if perturbations else np.array([])
    return accuracy, perturbation_array, adversarial_batch, clean_batch


def _plot_attacks_comparison(
    clean_accuracy: float,
    scratch_accuracy: float,
    art_accuracy: float,
    output_dir: Path,
) -> Path:
    """Plot accuracy comparison: clean vs FGSM from scratch vs FGSM with ART."""
    figure, axis = plt.subplots(figsize=(8, 5))
    methods = ["Clean", "FGSM (Scratch)", "FGSM (ART)"]
    accuracies = [clean_accuracy, scratch_accuracy, art_accuracy]
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    axis.bar(methods, accuracies, color=colors, alpha=0.7)
    axis.set_ylabel("Accuracy")
    axis.set_title("CIFAR-10 Accuracy: Clean vs FGSM Attacks")
    axis.set_ylim([0, 1])
    for i, (method, acc) in enumerate(zip(methods, accuracies)):
        axis.text(i, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=10)
    figure.tight_layout()
    path = output_dir / "fgsm_comparison.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_adversarial_samples(
    clean_images: torch.Tensor,
    adversarial_scratch: torch.Tensor,
    adversarial_art: torch.Tensor,
    output_dir: Path,
) -> Path:
    """Plot clean vs adversarial images side by side."""
    sample_count = min(10, clean_images.shape[0], adversarial_scratch.shape[0], adversarial_art.shape[0])
    figure, axes = plt.subplots(3, sample_count, figsize=(3.2 * sample_count, 9))
    figure.suptitle("Clean vs Adversarial Images (FGSM from scratch vs ART)")

    denorm = lambda x: (x + 1) / 2  # Undo normalization
    for i in range(sample_count):
        axes[0, i].imshow(denorm(clean_images[i]).permute(1, 2, 0).clamp(0, 1))
        axes[0, i].set_title("Clean")
        axes[0, i].axis("off")

        axes[1, i].imshow(denorm(adversarial_scratch[i]).permute(1, 2, 0).clamp(0, 1))
        axes[1, i].set_title("FGSM (Scratch)")
        axes[1, i].axis("off")

        axes[2, i].imshow(denorm(adversarial_art[i]).permute(1, 2, 0).clamp(0, 1))
        axes[2, i].set_title("FGSM (ART)")
        axes[2, i].axis("off")

    figure.tight_layout()
    path = output_dir / "adversarial_samples.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def run_q2_fgsm(
    training_config: TrainingConfig,
    attack_config: AttackConfig,
    experiment_name: str = "q2_fgsm",
    min_clean_accuracy: float = 0.72,
) -> dict[str, float | Path]:
    """Run Q2.1: Train clean model and compare FGSM attacks."""
    set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    output_dir = default_output_dir(training_config.output_dir, experiment_name)
    wandb_run = _init_wandb(
        training_config.wandb_project,
        training_config.wandb_entity,
        training_config.run_name,
        training_config.tags,
    )

    clean_model, clean_result = _train_resnet18_clean(training_config, output_dir, device)
    print(f"Clean accuracy: {clean_result.test_metrics['accuracy']:.4f}")
    if clean_result.test_metrics["accuracy"] < min_clean_accuracy:
        print(f"Warning: clean accuracy {clean_result.test_metrics['accuracy']:.4f} < {min_clean_accuracy}")

    if wandb_run is not None:
        wandb_run.config.update({
            "task": "fgsm",
            "batch_size": training_config.batch_size,
            "epochs": training_config.epochs,
            "epsilon": attack_config.epsilon,
        })

    data = build_cifar10_loaders(
        training_config.data_root,
        training_config.batch_size,
        training_config.num_workers,
        image_size=224,
    )

    def attack_from_scratch(model, inputs, targets):
        return fgsm_attack(model, inputs, targets, epsilon=attack_config.epsilon)

    scratch_acc, scratch_perturbations, adv_scratch, clean_batch = _evaluate_attack(
        clean_model, attack_from_scratch, data.test_loader, device
    )
    art_acc, art_perturbations, adv_art, _ = _evaluate_attack(
        clean_model,
        lambda m, i, t: art_fgsm_attack(m, i, t, attack_config.epsilon, num_classes=10),
        data.test_loader,
        device,
    )

    clean_result.history.to_csv(output_dir / "clean_training_history.csv", index=False)
    attack_results = pd.DataFrame({
        "method": ["clean", "fgsm_scratch", "fgsm_art"],
        "accuracy": [clean_result.test_metrics["accuracy"], scratch_acc, art_acc],
    })
    attack_results.to_csv(output_dir / "attack_results.csv", index=False)

    fgsm_comparison_path = _plot_attacks_comparison(
        clean_result.test_metrics["accuracy"],
        scratch_acc,
        art_acc,
        output_dir,
    )

    sample_path = None
    if adv_scratch is not None and adv_art is not None and clean_batch is not None:
        sample_path = _plot_adversarial_samples(clean_batch, adv_scratch, adv_art, output_dir)

    if wandb_run is not None:
        _log_wandb_image(wandb_run, "fgsm_comparison", fgsm_comparison_path)
        if sample_path is not None:
            _log_wandb_image(wandb_run, "fgsm_samples", sample_path)
        wandb_run.log({
            "clean_accuracy": clean_result.test_metrics["accuracy"],
            "fgsm_scratch_accuracy": scratch_acc,
            "fgsm_art_accuracy": art_acc,
        })

    return {
        "clean_accuracy": clean_result.test_metrics["accuracy"],
        "fgsm_scratch_accuracy": scratch_acc,
        "fgsm_art_accuracy": art_acc,
        "output_dir": output_dir,
    }


def run_q2_detector(
    training_config: TrainingConfig,
    attack_config: AttackConfig,
    attack_type: str = "pgd",
    experiment_name: str | None = None,
) -> dict[str, float | Path]:
    """Run Q2.2: Train adversarial detector (clean vs adversarial)."""
    if attack_type not in ("pgd", "bim"):
        raise ValueError(f"attack_type must be 'pgd' or 'bim', got {attack_type}")

    set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    if experiment_name is None:
        experiment_name = f"q2_detector_{attack_type}"
    output_dir = default_output_dir(training_config.output_dir, experiment_name)
    wandb_run = _init_wandb(
        training_config.wandb_project,
        training_config.wandb_entity,
        training_config.run_name,
        training_config.tags,
    )

    if wandb_run is not None:
        wandb_run.config.update({
            "task": "detect",
            "attack_type": attack_type,
            "batch_size": training_config.batch_size,
            "epochs": training_config.epochs,
            "epsilon": attack_config.epsilon,
            "alpha": attack_config.alpha,
            "steps": attack_config.steps,
        })

    data = build_cifar10_loaders(
        training_config.data_root,
        training_config.batch_size,
        training_config.num_workers,
        image_size=224,
    )

    clean_model = build_resnet18(num_classes=data.num_classes, pretrained=False)
    clean_model.to(device)

    optimizer_clean = torch.optim.SGD(
        clean_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    for epoch in range(1, training_config.epochs + 1):
        train_epoch(clean_model, data.train_loader, device, optimizer_clean)
    clean_model.eval()

    detector = build_resnet34_detector(num_classes=2, pretrained=False)
    detector.to(device)

    attack_fn = art_pgd_attack if attack_type == "pgd" else art_bim_attack

    detector_optimizer = torch.optim.SGD(
        detector.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    rows: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_detector_path = output_dir / f"best_{attack_type}_detector.pth"
    sample_clean_batch: torch.Tensor | None = None
    sample_adv_batch: torch.Tensor | None = None

    for epoch in range(1, training_config.epochs + 1):
        detector.train(True)
        total_correct = 0
        total = 0

        for inputs, targets in data.train_loader:
            inputs = inputs.to(device, non_blocking=True)

            adv = attack_fn(
                clean_model,
                inputs,
                targets,
                epsilon=attack_config.epsilon,
                alpha=attack_config.alpha,
                steps=attack_config.steps,
                num_classes=10,
            )
            adv = torch.clamp(adv, 0.0, 1.0)

            if sample_clean_batch is None:
                sample_clean_batch = inputs.detach().cpu()
                sample_adv_batch = adv.detach().cpu()

            clean_batch = torch.cat([inputs, adv], dim=0)
            labels_batch = torch.cat(
                [
                    torch.zeros(inputs.size(0), dtype=torch.long),
                    torch.ones(adv.size(0), dtype=torch.long),
                ],
                dim=0,
            ).to(device)

            detector_optimizer.zero_grad(set_to_none=True)
            outputs = detector(clean_batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = torch.nn.functional.cross_entropy(logits, labels_batch)
            loss.backward()
            detector_optimizer.step()

            total_correct += (logits.argmax(dim=1) == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_accuracy = total_correct / max(total, 1)

        detector.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in data.val_loader:
                inputs = inputs.to(device, non_blocking=True)
                with torch.enable_grad():
                    adv = attack_fn(
                        clean_model,
                        inputs,
                        targets,
                        epsilon=attack_config.epsilon,
                        alpha=attack_config.alpha,
                        steps=attack_config.steps,
                        num_classes=10,
                    )
                adv = torch.clamp(adv, 0.0, 1.0)

                if sample_clean_batch is None:
                    sample_clean_batch = inputs.detach().cpu()
                    sample_adv_batch = adv.detach().cpu()

                clean_batch = torch.cat([inputs, adv], dim=0)
                labels_batch = torch.cat(
                    [
                        torch.zeros(inputs.size(0), dtype=torch.long),
                        torch.ones(adv.size(0), dtype=torch.long),
                    ],
                    dim=0,
                ).to(device)

                outputs = detector(clean_batch)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                val_correct += (logits.argmax(dim=1) == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        val_accuracy = val_correct / max(val_total, 1)
        row = {"epoch": epoch, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy}
        rows.append(row)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(detector, output_dir, f"best_{attack_type}_detector.pth")

    detector.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in data.test_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.enable_grad():
                adv = attack_fn(
                    clean_model,
                    inputs,
                    targets,
                    epsilon=attack_config.epsilon,
                    alpha=attack_config.alpha,
                    steps=attack_config.steps,
                    num_classes=10,
                )
            adv = torch.clamp(adv, 0.0, 1.0)

            if sample_clean_batch is None:
                sample_clean_batch = inputs.detach().cpu()
                sample_adv_batch = adv.detach().cpu()

            clean_batch = torch.cat([inputs, adv], dim=0)
            labels_batch = torch.cat(
                [
                    torch.zeros(inputs.size(0), dtype=torch.long),
                    torch.ones(adv.size(0), dtype=torch.long),
                ],
                dim=0,
            ).to(device)

            outputs = detector(clean_batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            test_correct += (logits.argmax(dim=1) == labels_batch).sum().item()
            test_total += labels_batch.size(0)

    test_accuracy = test_correct / max(test_total, 1)
    history = pd.DataFrame(rows)
    history.to_csv(output_dir / f"{attack_type}_detector_history.csv", index=False)

    sample_path = None
    if sample_clean_batch is not None and sample_adv_batch is not None:
        sample_label = "PGD" if attack_type == "pgd" else "BIM"
        sample_path = _plot_sample_grid(sample_clean_batch, sample_adv_batch, sample_label, output_dir)

    if wandb_run is not None:
        wandb_run.log({"test_accuracy": test_accuracy})
        if sample_path is not None:
            _log_wandb_image(wandb_run, f"{attack_type}_samples", sample_path)

    return {
        "attack_type": attack_type,
        "test_accuracy": test_accuracy,
        "output_dir": output_dir,
        "checkpoint": str(best_detector_path),
    }
