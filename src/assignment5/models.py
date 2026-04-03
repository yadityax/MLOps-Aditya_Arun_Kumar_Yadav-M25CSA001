from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import logging as transformers_logging
from transformers import ViTForImageClassification


def build_vit_small(
    num_classes: int,
    pretrained_name: str = "WinKawaks/vit-small-patch16-224",
) -> ViTForImageClassification:
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        model = ViTForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
    return model


def attach_lora_to_vit(
    model: torch.nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...] = ("query", "key", "value"),
) -> torch.nn.Module:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
        task_type="SEQ_CLS",
    )
    return get_peft_model(model, lora_config)


def build_resnet18(num_classes: int = 10, pretrained: bool = False) -> torch.nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet34_detector(num_classes: int = 2, pretrained: bool = False) -> torch.nn.Module:
    from torchvision.models import ResNet34_Weights, resnet34

    weights = ResNet34_Weights.DEFAULT if pretrained else None
    model = resnet34(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
