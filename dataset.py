from collections import Counter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE
from transforms import data_transforms

def get_datasets():
    """ 
    Load the train, validation, and test datasets.
    """
    train_dset = ImageFolder(root=TRAIN_DIR, transform=data_transforms["train"])
    val_dset = ImageFolder(root=VAL_DIR, transform=data_transforms["val"])
    test_dset = ImageFolder(root=TEST_DIR, transform=data_transforms["test"])
    return train_dset, val_dset, test_dset

def get_dataloaders():
    """
    Create DataLoaders for train, val, and test.
    """
    train_dset, val_dset, test_dset = get_datasets()
    
    dataloaders = {
        'train': DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        'test': DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    return dataloaders

def count_classes(dset):
    """
    Count the number of samples per class in a dataset.
    """
    labels = [label for _, label in dset]
    return Counter(labels)

def get_dataset_stats():
    """
    Print out dataset statistics.
    """
    train_dset, val_dset, test_dset = get_datasets()
    
    train_counts = count_classes(train_dset)
    val_counts = count_classes(val_dset)
    test_counts = count_classes(test_dset)
    
    print("Train:", train_counts)
    print("Val:", val_counts)
    print("Test:", test_counts)
    print("Classes:", train_dset.classes)
    
    print("Train samples:", len(train_dset))
    print("Val samples:", len(val_dset))
    print("Test samples:", len(test_dset))
    
    return train_counts, val_counts, test_counts
