from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch

from assignment5.config import LoRAConfig, TrainingConfig
from assignment5.huggingface import save_and_push_to_huggingface
from assignment5.utils import to_dict
from assignment5.data import build_cifar100_loaders
from assignment5.engine import evaluate, extract_logits, save_checkpoint, train_epoch
from assignment5.lora import count_parameters, freeze_backbone_except_head
from assignment5.models import attach_lora_to_vit, build_vit_small
from assignment5.utils import default_output_dir, ensure_dir, resolve_device, set_seed


@dataclass(slots=True)
class Q1ExperimentResult:
    history: pd.DataFrame
    test_metrics: dict[str, float]
    classwise_accuracy: pd.Series
    checkpoint_path: Path
    output_dir: Path


def _build_model(
    num_classes: int,
    use_lora: bool,
    lora_config: LoRAConfig,
    pretrained_model_name: str,
) -> torch.nn.Module:
    model = build_vit_small(num_classes=num_classes, pretrained_name=pretrained_model_name)
    if use_lora:
        model = attach_lora_to_vit(
            model,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
        )
    else:
        freeze_backbone_except_head(model)
    return model


def _save_history(history: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "history.csv"
    history.to_csv(path, index=False)
    return path


def _save_required_train_val_table(history: pd.DataFrame, output_dir: Path) -> Path:
    required = history[["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"]].copy()
    required.columns = [
        "Epoch",
        "Training Loss",
        "Validation Loss",
        "Training Accuracy",
        "Validation Accuracy",
    ]
    path = output_dir / "q1_train_val_table_required.csv"
    required.to_csv(path, index=False)
    return path


def _save_required_sweep_train_val_table(
    table_rows: list[dict[str, float | int]],
    output_dir: Path,
) -> Path:
    required = pd.DataFrame(table_rows)
    path = output_dir / "q1_train_val_table_required_all_experiments.csv"
    required.to_csv(path, index=False)
    return path


def _plot_curves(history: pd.DataFrame, output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    path = output_dir / "training_curves.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_classwise_accuracy(classwise_accuracy: pd.Series, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(14, 5))
    classwise_accuracy.sort_index().plot(kind="bar", ax=axis, color="#3b82f6")
    axis.set_xlabel("Class")
    axis.set_ylabel("Accuracy")
    axis.set_title("Class-wise Test Accuracy")
    figure.tight_layout()
    path = output_dir / "classwise_test_accuracy.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_gradient_norms(history: pd.DataFrame, output_dir: Path) -> Path | None:
    gradient_columns = [column for column in history.columns if column.startswith("grad_")]
    if not gradient_columns:
        return None

    figure, axis = plt.subplots(figsize=(10, 5))
    for column in gradient_columns:
        axis.plot(history["epoch"], history[column], label=column.removeprefix("grad_"))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Average gradient norm")
    axis.set_title("LoRA Gradient Norms During Training")
    axis.legend()
    figure.tight_layout()
    path = output_dir / "lora_gradient_norms.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _init_wandb(project: str, entity: str | None, run_name: str | None, tags: list[str] | None):
    try:
        import wandb
    except ImportError:  # pragma: no cover - optional dependency
        return None
    wandb.init(project=project, entity=entity, name=run_name, tags=tags or [])
    return wandb


def _log_history_to_wandb(
    wandb,
    history: pd.DataFrame,
    output_dir: Path,
    experiment_name: str,
):
    if wandb is None:
        return
    for _, row in history.iterrows():
        wandb.log({
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "train_accuracy": row["train_accuracy"],
            "val_loss": row["val_loss"],
            "val_accuracy": row["val_accuracy"],
        })
    wandb.log({"training_curves": wandb.Image(str(output_dir / "training_curves.png"))})
    wandb.log({"classwise_accuracy": wandb.Image(str(output_dir / "classwise_test_accuracy.png"))})
    gradient_path = output_dir / "lora_gradient_norms.png"
    if gradient_path.exists():
        wandb.log({"lora_gradients": wandb.Image(str(gradient_path))})


def _classwise_accuracy(model: torch.nn.Module, loader: Iterable, device: torch.device, num_classes: int) -> pd.Series:
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.float64)
    total = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(pixel_values=inputs)
            logits = extract_logits(outputs)
            predictions = logits.argmax(dim=1)
            for class_index in range(num_classes):
                class_mask = targets == class_index
                total[class_index] += class_mask.sum().item()
                correct[class_index] += ((predictions == targets) & class_mask).sum().item()

    accuracy = correct / total.clamp_min(1)
    return pd.Series(accuracy.cpu().numpy(), index=[f"class_{index}" for index in range(num_classes)], name="accuracy")


def run_q1_experiment(
    *,
    use_lora: bool,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    experiment_name: str,
    save_artifacts: bool = True,
    log_wandb: bool = False,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
) -> Q1ExperimentResult:
    set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    data = build_cifar100_loaders(
        training_config.data_root,
        training_config.batch_size,
        training_config.num_workers,
        training_config.image_size,
    )
    model = _build_model(
        data.num_classes,
        use_lora,
        lora_config,
        training_config.pretrained_model_name,
    )
    model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    output_dir = default_output_dir(training_config.output_dir, experiment_name)
    report = count_parameters(model)
    wandb_run = _init_wandb(training_config.wandb_project, training_config.wandb_entity, training_config.run_name, training_config.tags) if log_wandb else None
    if wandb_run is not None:
        wandb_run.config.update(to_dict(training_config))
        wandb_run.config.update(to_dict(lora_config) if use_lora else {"lora": False})
        wandb_run.config.update({
            "total_parameters": report.total_parameters,
            "trainable_parameters": report.trainable_parameters,
            "trainable_ratio": report.trainable_ratio,
        })

    rows: list[dict[str, float | int | str]] = []
    best_checkpoint_path = output_dir / "best_model.pth"
    best_val_accuracy = -1.0

    gradient_prefixes = ("lora_",) if use_lora else ()
    for epoch in range(1, training_config.epochs + 1):
        train_result = train_epoch(model, data.train_loader, device, optimizer, gradient_prefixes=gradient_prefixes)
        val_metrics = evaluate(model, data.val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_result.metrics.loss,
            "train_accuracy": train_result.metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "trainable_parameters": report.trainable_parameters,
            "total_parameters": report.total_parameters,
        }
        for prefix, gradient_value in train_result.gradient_norms.items():
            row[f"grad_{prefix}"] = gradient_value
        rows.append(row)
        if val_metrics.accuracy >= best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            save_checkpoint(model, output_dir, "best_model.pth")

    history = pd.DataFrame(rows)
    test_metrics = evaluate(model, data.test_loader, device)
    classwise_accuracy = _classwise_accuracy(model, data.test_loader, device, data.num_classes)

    if save_artifacts:
        _save_history(history, output_dir)
        _save_required_train_val_table(history, output_dir)
        _plot_curves(history, output_dir)
        _plot_classwise_accuracy(classwise_accuracy, output_dir)
        _plot_gradient_norms(history, output_dir)
        classwise_accuracy.to_csv(output_dir / "classwise_test_accuracy.csv")
        pd.DataFrame([{"loss": test_metrics.loss, "accuracy": test_metrics.accuracy}]).to_csv(
            output_dir / "test_metrics.csv",
            index=False,
        )
        if wandb_run is not None:
            _log_history_to_wandb(wandb_run, history, output_dir, experiment_name)
            wandb_run.log({"test_loss": test_metrics.loss, "test_accuracy": test_metrics.accuracy})
        if hf_repo_id:
            try:
                hf_url = save_and_push_to_huggingface(model, best_checkpoint_path, hf_repo_id, hf_token=hf_token)
                print(f"Model pushed to {hf_url}")
                if wandb_run is not None:
                    wandb_run.log({"huggingface_url": hf_url})
            except Exception as exc:  # pragma: no cover
                print(f"Warning: Could not export to HuggingFace: {exc}")

    return Q1ExperimentResult(
        history=history,
        test_metrics={"loss": test_metrics.loss, "accuracy": test_metrics.accuracy},
        classwise_accuracy=classwise_accuracy,
        checkpoint_path=best_checkpoint_path,
        output_dir=output_dir,
    )


def run_q1_sweep(
    training_config: TrainingConfig,
    ranks: tuple[int, ...] = (2, 4, 8),
    alphas: tuple[int, ...] = (2, 4, 8),
    dropout: float = 0.1,
) -> pd.DataFrame:
    results: list[dict[str, float | int | str]] = []
    train_val_required_rows: list[dict[str, float | int]] = []
    experiment_no = 0
    for rank, alpha in product(ranks, alphas):
        experiment_no += 1
        config = LoRAConfig(rank=rank, alpha=alpha, dropout=dropout)
        experiment_name = f"q1_lora_r{rank}_a{alpha}_d{dropout}"
        result = run_q1_experiment(
            use_lora=True,
            lora_config=config,
            training_config=training_config,
            experiment_name=experiment_name,
            save_artifacts=True,
        )

        for _, history_row in result.history.iterrows():
            train_val_required_rows.append(
                {
                    "Experiment No. (based on combinations in step 2)": experiment_no,
                    "Rank": rank,
                    "Alpha": alpha,
                    "Epoch": int(history_row["epoch"]),
                    "Training Loss": float(history_row["train_loss"]),
                    "Validation Loss": float(history_row["val_loss"]),
                    "Training Accuracy": float(history_row["train_accuracy"]),
                    "Validation Accuracy": float(history_row["val_accuracy"]),
                }
            )

        results.append(
            {
                "experiment": experiment_name,
                "LoRA layers (with/without)": "With LoRA (Q,K,V)",
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "overall_test_accuracy": result.test_metrics["accuracy"],
                "trainable_parameters_used": int(result.history["trainable_parameters"].iloc[0]),
                "best_checkpoint": str(result.checkpoint_path),
            }
        )
    summary = pd.DataFrame(results)
    ensure_dir(training_config.output_dir)
    summary.to_csv(Path(training_config.output_dir) / "q1_lora_sweep_summary.csv", index=False)

    required_table = summary[
        [
            "LoRA layers (with/without)",
            "rank",
            "alpha",
            "dropout",
            "overall_test_accuracy",
            "trainable_parameters_used",
        ]
    ].copy()
    required_table.columns = [
        "LoRA layers (with/without)",
        "Rank",
        "Alpha",
        "Dropout",
        "Overall Test Accuracy",
        "Trainable Parameters used",
    ]

    baseline_metrics_path = Path(training_config.output_dir) / "q1_baseline" / "test_metrics.csv"
    baseline_history_path = Path(training_config.output_dir) / "q1_baseline" / "history.csv"
    if baseline_metrics_path.exists() and baseline_history_path.exists():
        baseline_metrics = pd.read_csv(baseline_metrics_path)
        baseline_history = pd.read_csv(baseline_history_path)
        baseline_row = pd.DataFrame(
            [
                {
                    "LoRA layers (with/without)": "Without LoRA",
                    "Rank": "-",
                    "Alpha": "-",
                    "Dropout": "-",
                    "Overall Test Accuracy": float(baseline_metrics["accuracy"].iloc[0]),
                    "Trainable Parameters used": int(baseline_history["trainable_parameters"].iloc[0]),
                }
            ]
        )
        required_table = pd.concat([baseline_row, required_table], ignore_index=True)

    _save_required_sweep_train_val_table(train_val_required_rows, Path(training_config.output_dir))
    required_table.to_csv(Path(training_config.output_dir) / "q1_testing_table_required.csv", index=False)
    return summary


def run_q1_optuna_objective(training_config: TrainingConfig, trial) -> float:
    rank = trial.suggest_categorical("rank", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [2, 4, 8])
    dropout = trial.suggest_categorical("dropout", [0.1])
    result = run_q1_experiment(
        use_lora=True,
        lora_config=LoRAConfig(rank=rank, alpha=alpha, dropout=dropout),
        training_config=training_config,
        experiment_name=f"q1_optuna_trial_{trial.number}",
        save_artifacts=False,
    )
    return result.test_metrics["accuracy"]
