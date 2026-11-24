"""Single-process OS-Fed components for the PDC project.

Snapshot initialization is a project choice (decision D4, "not stated" in the
paper's main text). This file supports two options so Step 8.10 can compare them:
  * snapshot_init: first_client_image  -> the snapshot starts from one real,
    normalized client image (one monolithic feature map, as in the paper figure).
  * snapshot_init: random              -> the snapshot starts from random noise.
Record the chosen option and the reason in notes/decisions.md after Step 8.10.
Verify against the OS-Fed supplementary PDF before calling this a reproduction.
"""

from __future__ import annotations

import time
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from src.client import Client, State, train_local
from src.data import normalize_batch


def trainable_vector(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten trainable parameters in the model's fixed parameter order."""
    values = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        tensor = state[name]
        if not tensor.is_floating_point():
            raise TypeError(f"Expected floating-point parameter: {name}")
        values.append(tensor.detach().reshape(-1).cpu())

    if not values:
        raise ValueError("The model has no trainable parameters")

    return torch.cat(values)


def local_delta(
    model: nn.Module,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return Δw = weights before local training − weights after training."""
    before = trainable_vector(model, global_state)
    after = trainable_vector(model, local_state)

    if before.shape != after.shape:
        raise ValueError("Global and local parameter vectors have different sizes")

    return before - after


SNAPSHOT_INIT_CHOICES = ("first_client_image", "random")


def make_initial_image(
    snapshot_init: str,
    client: Client,
    shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build the starting feature map for a snapshot (decision D4).

    Returns a tensor of shape [1, channels, height, width] on `device`.
    """
    channels, height, width = shape

    if snapshot_init == "first_client_image":
        image = normalize_batch(client.loader.x[:1]).to(device=device, dtype=dtype)
        if tuple(image.shape[1:]) != (channels, height, width):
            raise ValueError(
                f"First client image is {tuple(image.shape[1:])}, "
                f"but snapshot_shape is {(channels, height, width)}"
            )
        return image.detach().clone()

    if snapshot_init == "random":
        # Build on CPU with a seeded generator so the random start is reproducible,
        # then move to the target device.
        noise = torch.randn(
            1, channels, height, width,
            dtype=dtype,
            generator=generator,
        )
        return noise.to(device=device)

    raise ValueError(
        f"Unknown snapshot_init {snapshot_init!r}; "
        f"choose one of {SNAPSHOT_INIT_CHOICES}"
    )


class ClientErrorFeedback:
    """Keep one error vector per client across communication rounds."""

    def __init__(self) -> None:
        self.errors: dict[int, torch.Tensor] = {}

    def target(self, client_id: int, delta_w: torch.Tensor) -> torch.Tensor:
        """Return Δg = Δw + the client's previous error."""
        previous = self.errors.get(client_id)
        if previous is None:
            previous = torch.zeros_like(delta_w)
        else:
            previous = previous.to(device=delta_w.device, dtype=delta_w.dtype)

        if previous.shape != delta_w.shape:
            raise ValueError(f"Client {client_id}: saved error has the wrong shape")

        return delta_w + previous

    def update(
        self,
        client_id: int,
        target_delta: torch.Tensor,
        synthetic_gradient_value: torch.Tensor,
    ) -> None:
        """Save e = Δg − synthetic gradient for the next round."""
        synthetic_gradient_value = synthetic_gradient_value.to(
            device=target_delta.device,
            dtype=target_delta.dtype,
        )
        if target_delta.shape != synthetic_gradient_value.shape:
            raise ValueError("Target and synthetic gradients have different sizes")

        self.errors[client_id] = (
            target_delta - synthetic_gradient_value
        ).detach().cpu().clone()

    def clear(self, client_id: int, like: torch.Tensor) -> None:
        """Keep the client's error at zero when error compensation is disabled."""
        self.errors[client_id] = torch.zeros_like(like).detach().cpu()

    def state_dict(self) -> dict[int, torch.Tensor]:
        """Return errors so the training checkpoint can save them."""
        return {cid: error.clone() for cid, error in self.errors.items()}

    def load_state_dict(self, state: Mapping[int, torch.Tensor]) -> None:
        """Restore errors when resuming a run."""
        self.errors = {
            int(cid): error.detach().cpu().clone()
            for cid, error in state.items()
        }


def grid_samples(
    image: torch.Tensor,
    label_logits: torch.Tensor,
    grid_size: int,
    output_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn one feature map into grid_size² resized images and soft labels.

    image shape:        [1, channels, height, width]
    label_logits shape: [grid_size², number_of_classes]
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape [1, channels, height, width]")
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")

    _, _, height, width = image.shape
    if height % grid_size or width % grid_size:
        raise ValueError("Image height and width must divide evenly into the grid")

    expected_labels = grid_size * grid_size
    if label_logits.shape[0] != expected_labels:
        raise ValueError(
            f"Expected {expected_labels} label vectors, got {label_logits.shape[0]}"
        )

    patch_h = height // grid_size
    patch_w = width // grid_size
    patches = []

    for row in range(grid_size):
        for col in range(grid_size):
            patch = image[
                :,
                :,
                row * patch_h:(row + 1) * patch_h,
                col * patch_w:(col + 1) * patch_w,
            ]
            patch = F.interpolate(
                patch,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )
            patches.append(patch)

    synthetic_images = torch.cat(patches, dim=0)
    soft_labels = torch.softmax(label_logits, dim=1)
    return synthetic_images, soft_labels


def synthetic_gradient(
    model: nn.Module,
    image: torch.Tensor,
    label_logits: torch.Tensor,
    grid_size: int,
    create_graph: bool,
) -> torch.Tensor:
    """Calculate the model gradient produced by a synthetic snapshot."""
    samples, soft_labels = grid_samples(image, label_logits, grid_size)

    predictions = model(samples)
    loss = -(
        soft_labels * F.log_softmax(predictions, dim=1)
    ).sum(dim=1).mean()

    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=create_graph,
    )
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def optimize_snapshot(
    model: nn.Module,
    image: nn.Parameter,
    label_logits: nn.Parameter,
    target_delta: torch.Tensor,
    grid_size: int,
    max_iter: int,
    lbfgs_lr: float,
    eps: float,
) -> dict[str, object]:
    """Use L-BFGS to match the snapshot gradient to target_delta."""
    first_parameter = next(model.parameters())
    target = target_delta.detach().to(
        device=first_parameter.device,
        dtype=first_parameter.dtype,
    )

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if image.device != first_parameter.device:
        raise ValueError("Snapshot image and model must be on the same device")
    if label_logits.device != first_parameter.device:
        raise ValueError("Snapshot labels and model must be on the same device")

    optimizer = torch.optim.LBFGS(
        [image, label_logits],
        lr=lbfgs_lr,
        max_iter=max_iter,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        generated = synthetic_gradient(
            model,
            image,
            label_logits,
            grid_size,
            create_graph=True,
        )
        if generated.shape != target.shape:
            raise ValueError(
                f"Snapshot gradient has {generated.numel()} values, "
                f"but target has {target.numel()}"
            )

        match_loss = (generated - target).square().sum()
        match_loss = match_loss / (target.square().sum() + eps)
        match_loss.backward()
        return match_loss

    optimizer.step(closure)

    model.zero_grad(set_to_none=True)
    final_gradient = synthetic_gradient(
        model,
        image,
        label_logits,
        grid_size,
        create_graph=False,
    ).detach()

    residual = target - final_gradient
    match_loss_value = (
        residual.square().sum() / (target.square().sum() + eps)
    ).item()
    cosine = F.cosine_similarity(
        target.reshape(1, -1),
        final_gradient.reshape(1, -1),
        dim=1,
        eps=eps,
    ).item()
    residual_ratio = (
        residual.norm() / (target.norm() + eps)
    ).item()

    return {
        "image": image.detach().cpu().clone(),
        "label_logits": label_logits.detach().cpu().clone(),
        "synthetic_gradient": final_gradient.detach().cpu().clone(),
        "match_loss": match_loss_value,
        "cosine_similarity": cosine,
        "residual_ratio": residual_ratio,
    }


class OSFed:
    """Single-process OS-Fed adapter with the same interface as FedAvg."""

    name = "osfed"

    def __init__(
        self,
        cfg: dict,
        work_model: nn.Module,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.work_model = work_model
        self.device = device
        self.error_feedback = ClientErrorFeedback()

        required = (
            "grid_size",
            "lbfgs_max_iter",
            "lbfgs_lr",
            "snapshot_eps",
            "snapshot_init",
            "error_compensation",
        )
        missing = [key for key in required if key not in cfg]
        if missing:
            raise ValueError(f"OS-Fed config is missing settings: {missing}")

        if cfg["snapshot_init"] not in SNAPSHOT_INIT_CHOICES:
            raise ValueError(
                f"snapshot_init must be one of {SNAPSHOT_INIT_CHOICES}, "
                f"got {cfg['snapshot_init']!r}"
            )

        shape = cfg.get("snapshot_shape", [3, 32, 32])
        if len(shape) != 3:
            raise ValueError("snapshot_shape must be [channels, height, width]")
        self.snapshot_shape = (int(shape[0]), int(shape[1]), int(shape[2]))

        # Reproducible random init (used only when snapshot_init == "random").
        self._rng = torch.Generator()
        self._rng.manual_seed(int(cfg.get("seed", 0)) * 7919 + 13)

    def state_dict(self) -> dict[int, torch.Tensor]:
        """Return client error vectors for checkpointing."""
        return self.error_feedback.state_dict()

    def load_state_dict(self, state: Mapping[int, torch.Tensor]) -> None:
        """Restore client error vectors from a checkpoint."""
        self.error_feedback.load_state_dict(state)

    def client_update(
        self,
        client: Client,
        global_state: State,
        lr: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Train locally, optimize a snapshot, and return the snapshot."""
        local_state, train_loss, train_seconds = train_local(
            self.work_model,
            global_state,
            client.loader,
            epochs=int(self.cfg["local_epochs"]),
            lr=float(lr),
            momentum=float(self.cfg["momentum"]),
            weight_decay=float(self.cfg["weight_decay"]),
            device=self.device,
        )

        delta_w = local_delta(self.work_model, global_state, local_state)
        target_delta = self.error_feedback.target(client.client_id, delta_w)

        # Snapshot matching uses the current global model weights.
        self.work_model.load_state_dict(global_state)
        self.work_model.to(self.device)
        self.work_model.eval()

        dtype = next(self.work_model.parameters()).dtype
        image_init = make_initial_image(
            str(self.cfg["snapshot_init"]),
            client,
            self.snapshot_shape,
            self.device,
            dtype,
            generator=self._rng,
        )
        image = nn.Parameter(image_init.detach().clone())

        grid_size = int(self.cfg["grid_size"])
        num_classes = int(self.cfg.get("num_classes", 10))
        label_logits = nn.Parameter(
            torch.zeros(
                grid_size * grid_size,
                num_classes,
                device=self.device,
                dtype=dtype,
            )
        )

        start = time.perf_counter()
        result = optimize_snapshot(
            model=self.work_model,
            image=image,
            label_logits=label_logits,
            target_delta=target_delta,
            grid_size=grid_size,
            max_iter=int(self.cfg["lbfgs_max_iter"]),
            lbfgs_lr=float(self.cfg["lbfgs_lr"]),
            eps=float(self.cfg["snapshot_eps"]),
        )
        snapshot_seconds = time.perf_counter() - start

        if bool(self.cfg["error_compensation"]):
            self.error_feedback.update(
                client.client_id,
                target_delta,
                result["synthetic_gradient"],
            )
        else:
            self.error_feedback.clear(client.client_id, target_delta)

        message = {
            "image": result["image"],
            "label_logits": result["label_logits"],
        }
        info = {
            "train_loss": float(train_loss),
            "local_train_s": float(train_seconds),
            "snapshot_opt_s": float(snapshot_seconds),
            "match_loss_final": float(result["match_loss"]),
            "cos_sim": float(result["cosine_similarity"]),
            "residual_ratio": float(result["residual_ratio"]),
        }
        return message, info

    def server_update(
        self,
        server,
        messages: list[dict[str, torch.Tensor]],
        counts: list[int],
    ) -> None:
        """Apply the mean snapshot gradient to the current global model."""
        del counts  # Each client sends the same number of grid samples.

        all_images = []
        all_labels = []
        grid_size = int(self.cfg["grid_size"])

        for message in messages:
            images, labels = grid_samples(
                message["image"].to(self.device),
                message["label_logits"].to(self.device),
                grid_size=grid_size,
            )
            all_images.append(images)
            all_labels.append(labels)

        images = torch.cat(all_images, dim=0)
        labels = torch.cat(all_labels, dim=0)

        model = server.model
        model.eval()
        predictions = model(images)
        loss = -(
            labels * F.log_softmax(predictions, dim=1)
        ).sum(dim=1).mean()

        parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        gradients = torch.autograd.grad(loss, parameters)

        # Paper's update: w_next = w_current − mean snapshot gradient.
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=-1.0)
