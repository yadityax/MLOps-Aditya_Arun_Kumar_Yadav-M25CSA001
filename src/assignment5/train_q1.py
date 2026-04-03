from __future__ import annotations

import argparse
from importlib import import_module
import warnings

from assignment5.config import LoRAConfig, TrainingConfig
from assignment5.q1 import run_q1_experiment, run_q1_optuna_objective, run_q1_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q1 ViT on CIFAR-100 with optional LoRA.")
    parser.add_argument("--mode", choices=["baseline", "lora", "sweep", "optuna"], default="baseline")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes. Use 0 in Docker unless shm is increased.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="WinKawaks/vit-small-patch16-224",
        help="HuggingFace model id for pre-trained ViT-S checkpoint.",
    )
    parser.add_argument("--hf-repo", type=str, default=None, help="HuggingFace repo ID for model upload.")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token.")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*use_return_dict.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*dtype\(\): align should be passed as Python or NumPy boolean.*",
        category=DeprecationWarning,
    )
    args = parse_args()
    config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        pretrained_model_name=args.pretrained_model,
    )
    if args.mode == "sweep":
        summary = run_q1_sweep(config, dropout=args.dropout)
        print(summary.to_string(index=False))
        return

    if args.mode == "optuna":
        try:
            optuna = import_module("optuna")
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("Optuna is required for --mode optuna") from exc

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: run_q1_optuna_objective(config, trial), n_trials=args.trials)
        print(f"Best trial: {study.best_trial.number} | value={study.best_value:.4f} | params={study.best_params}")
        return

    use_lora = args.mode == "lora"
    result = run_q1_experiment(
        use_lora=use_lora,
        lora_config=LoRAConfig(rank=args.rank, alpha=args.alpha, dropout=args.dropout),
        training_config=config,
        experiment_name="q1_lora" if use_lora else "q1_baseline",
        save_artifacts=True,
        log_wandb=True,
        hf_repo_id=args.hf_repo,
        hf_token=args.hf_token,
    )
    print(result.history.to_string(index=False))
    print(f"Test metrics: {result.test_metrics}")


if __name__ == "__main__":
    main()
