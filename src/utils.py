import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def get_device() -> torch.device:
    """Return the best available compute device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, filename: str = "checkpoint.pth.tar") -> None:
    """Save model + optimizer state as a checkpoint."""
    print("=> Saving checkpoint")
    torch.save(state, filename)


def save_model(model: torch.nn.Module, path: str) -> None:
    """Persist model state dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """Load model weights from disk and move to device."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def plot_curves(
    train_losses, val_losses,
    train_accs,   val_accs,
    save_path: str = None,
) -> None:
    """Plot and optionally save training / validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs,   label="Val Acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Curves saved to {save_path}")
    else:
        plt.show()
    plt.close()
