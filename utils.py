import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def get_device():
    """Returns the available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model and optimizer state."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def save_model(model, path):
    """Save model state dict."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """Load model state dict."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation curves."""
    plt.figure(figsize=(10, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Curves saved to {save_path}")
    else:
        plt.show()
    plt.close()
