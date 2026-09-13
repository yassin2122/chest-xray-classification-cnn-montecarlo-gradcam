"""
src/dataset.py – Dataset loading and DataLoader construction.
"""
from collections import Counter

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from .config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE
from .transforms import data_transforms


def get_datasets():
    """Load ImageFolder datasets for train, val, and test splits."""
    train_dset = ImageFolder(root=TRAIN_DIR, transform=data_transforms["train"])
    val_dset   = ImageFolder(root=VAL_DIR,   transform=data_transforms["val"])
    test_dset  = ImageFolder(root=TEST_DIR,  transform=data_transforms["test"])
    return train_dset, val_dset, test_dset


def get_dataloaders(num_workers: int = 0):
    """Build DataLoaders for train, val, and test splits."""
    train_dset, val_dset, test_dset = get_datasets()
    return {
        "train": DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers),
        "val":   DataLoader(val_dset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers),
        "test":  DataLoader(test_dset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers),
    }


def count_classes(dset) -> Counter:
    """Return a Counter mapping class index → sample count."""
    return Counter(label for _, label in dset)


def get_dataset_stats():
    """Print and return class-level statistics for every split."""
    train_dset, val_dset, test_dset = get_datasets()
    stats = {
        "train": count_classes(train_dset),
        "val":   count_classes(val_dset),
        "test":  count_classes(test_dset),
    }
    for split, counts in stats.items():
        print(f"{split:>5} → {dict(counts)}  (total: {sum(counts.values())})")
    print("Classes:", train_dset.classes)
    return stats
