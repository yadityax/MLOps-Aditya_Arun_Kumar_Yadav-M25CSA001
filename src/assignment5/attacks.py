from __future__ import annotations

import torch


def fgsm_attack(model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    model.eval()
    inputs = inputs.detach().clone().requires_grad_(True)
    outputs = model(inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    loss = torch.nn.functional.cross_entropy(logits, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    perturbed = inputs + epsilon * inputs.grad.sign()
    return torch.clamp(perturbed, 0.0, 1.0).detach()


def iterative_fgsm_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    adv = inputs.detach().clone()
    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)

    for _ in range(steps):
        adv.requires_grad_(True)
        outputs = model(adv)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = torch.nn.functional.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        delta = torch.clamp(adv - inputs, min=-epsilon, max=epsilon)
        adv = torch.clamp(inputs + delta, 0.0, 1.0).detach()
    return adv


def _make_art_classifier(
    model: torch.nn.Module,
    num_classes: int,
    clip_values: tuple[float, float] = (0.0, 1.0),
):
    try:
        import numpy as np
        from art.estimators.classification import PyTorchClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("ART is not installed. Install adversarial-robustness-toolbox to use ART attacks.") from exc

    dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
    dummy_label = np.zeros((1, num_classes), dtype=np.float32)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=dummy_input.shape[1:],
        nb_classes=dummy_label.shape[1],
        clip_values=clip_values,
    )


def art_fgsm_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    num_classes: int = 10,
) -> torch.Tensor:
    classifier = _make_art_classifier(model, num_classes=num_classes)
    from art.attacks.evasion import FastGradientMethod

    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    adversarial = attack.generate(
        x=inputs.detach().cpu().numpy(),
        y=torch.nn.functional.one_hot(labels, num_classes=num_classes).cpu().numpy(),
    )
    return torch.from_numpy(adversarial).to(inputs.device)


def art_pgd_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    num_classes: int = 10,
) -> torch.Tensor:
    classifier = _make_art_classifier(model, num_classes=num_classes)
    from art.attacks.evasion import ProjectedGradientDescent

    attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=alpha, max_iter=steps)
    adversarial = attack.generate(
        x=inputs.detach().cpu().numpy(),
        y=torch.nn.functional.one_hot(labels, num_classes=num_classes).cpu().numpy(),
    )
    return torch.from_numpy(adversarial).to(inputs.device)


def art_bim_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    num_classes: int = 10,
) -> torch.Tensor:
    classifier = _make_art_classifier(model, num_classes=num_classes)
    from art.attacks.evasion import BasicIterativeMethod

    attack = BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=alpha, max_iter=steps)
    adversarial = attack.generate(
        x=inputs.detach().cpu().numpy(),
        y=torch.nn.functional.one_hot(labels, num_classes=num_classes).cpu().numpy(),
    )
    return torch.from_numpy(adversarial).to(inputs.device)
