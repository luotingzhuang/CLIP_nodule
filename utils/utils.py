import torch
import numpy as np
from models.CLIP import CLIPModel
from loralib.utils import apply_lora, mark_only_lora_as_trainable
from dataset.dataset_text import CLIPDatasetText, CLIPDatasetTextCollator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AvgMeter:
    """
    Average meter for tracking metrics.
    """

    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class EarlyStopping:
    """
    Early stopping mechanism to stop training when validation loss stops improving.
    """

    def __init__(
        self, warmup: int = 0, patience: int = 20, verbose: bool = True
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.lowest_val_loss = np.Inf
        self.early_stop = False
        self.warmup = warmup

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        model: torch.nn.Module,
        ckpt_path: str = "checkpoint.pt",
    ) -> None:
        if epoch < self.warmup:
            pass
        elif np.isinf(self.lowest_val_loss):
            self.save_checkpoint(val_loss, model, ckpt_path)
            self.lowest_val_loss = val_loss
        elif val_loss > self.lowest_val_loss:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, ckpt_path)
            self.lowest_val_loss = val_loss
            self.counter = 0

        return self.early_stop

    def save_checkpoint(
        self, val_loss: float, model: torch.nn.Module, ckpt_path: str
    ) -> None:
        if self.verbose:
            print(
                f"Validation loss decreased from {self.lowest_val_loss:.6f} to {val_loss:.6f}. Model saved."
            )
        torch.save({"model": model.state_dict()}, ckpt_path)


def init_model(args):
    """
    Initialize the model based on the provided arguments.
    Args:
        args: Arguments loaded from the saved model
    Returns:
        model: CLIP model
    """
    model = CLIPModel(args)

    if "ft" in args.tuning:
        print("Full tuning")
    elif args.tuning == "pt":
        print("Probe tuning")
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if (
                "image_projection" in name
                or "semantic_projection" in name
                or "classifier_image" in name
                or "classifier_text" in name
                or "logit_scale" in name
            ):
                param.requires_grad = True

    elif args.tuning == "lora":
        print("LoRA tuning")
        args.backbone = args.model.split("_")[1]
        list_lora_layers = apply_lora(args, model)
        mark_only_lora_as_trainable(model)

        for name, param in model.named_parameters():
            if (
                "image_projection" in name
                or "semantic_projection" in name
                or "classifier_image" in name
                or "classifier_text" in name
                or "logit_scale" in name
            ):
                param.requires_grad = True

    model = model.to(device)
    return model


def build_loaders(args, fold, mode):
    """
    Build data loaders for training, validation.
    Args:
        args: Arguments containing the configuration for the dataset.
        fold: Fold number for cross-validation.
        mode: Mode of the dataset ('train', 'val').
    Returns:
        dataloader: DataLoader for the specified mode.
    """

    dataset = CLIPDatasetText(args, fold, mode=mode, seed=0)
    train_collate_fn = CLIPDatasetTextCollator(args, mode=mode)

    if args.weighted == "diagnosis":
        sample_weights = dataset.sample_weights
    elif args.weighted == "semantic":
        sample_weights = dataset.semantic_weights
    else:
        raise ValueError("Invalid weighted argument")

    if mode == "train":
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(dataset), replacement=True
        )
        args.class_weights = dataset.class_weights
        print(f"Class weights: {args.class_weights}")
    else:
        sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        collate_fn=train_collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=False,
    )
    return dataloader
