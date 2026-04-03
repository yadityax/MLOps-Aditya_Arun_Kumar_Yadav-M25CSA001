from __future__ import annotations

import argparse
import warnings

from assignment5.config import AttackConfig, TrainingConfig
from assignment5.q2 import run_q2_detector, run_q2_fgsm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q2 experiments: FGSM attacks or adversarial detection.")
    parser.add_argument("--task", choices=["fgsm", "detect"], default="fgsm")
    parser.add_argument("--attack-type", choices=["pgd", "bim"], default="pgd", help="Type of attack for detection task.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes. Use 0 in Docker unless shm is increased.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*dtype\(\): align should be passed as Python or NumPy boolean.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*use_return_dict.*deprecated.*",
        category=FutureWarning,
    )
    args = parse_args()
    training_config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
    )
    attack_config = AttackConfig(
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps,
    )

    if args.task == "fgsm":
        result = run_q2_fgsm(training_config, attack_config)
        print(result)
    else:
        result = run_q2_detector(training_config, attack_config, attack_type=args.attack_type)
        print(f"Detector for {args.attack_type}: test_accuracy={result['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
