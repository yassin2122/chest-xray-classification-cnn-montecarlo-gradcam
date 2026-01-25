import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import sys
from tqdm import tqdm

# Add src to PYTHONPATH (in case run from root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import NUM_EPOCHS, LEARNING_RATE, MODELS_DIR, PATIENCE, RESULTS_DIR, DEVICE
from dataset import get_dataloaders
from model import CNN_MLP_Model, CNN_GAP_Model
from utils import get_device, save_model, plot_curves


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    since = time.time()
    device = get_device()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # Tracks best validation accuracy

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Run both training and validation phases
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(dataloaders[phase], desc=f"{phase} Phase")

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            # Track best model based on validation accuracy
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0

                    # Save best model immediately
                    save_model(model, os.path.join(MODELS_DIR, 'best_model_v2.pth'))
                else:
                    epochs_no_improve += 1

            # Scheduler step on validation loss (after each epoch)
            if phase == 'val' and scheduler is not None:
                scheduler.step(epoch_loss)

        print()

        # Early stopping trigger (based on validation performance)
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs!')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    plot_curves(
        train_losses, val_losses,
        train_accs, val_accs,
        save_path=os.path.join(RESULTS_DIR, 'training_curves.png')
    )

    return model

def compute_class_weights_from_dataset(dset):
    # ImageFolder has .targets list of class indices
    import torch
    targets = torch.tensor(dset.targets)
    num_classes = targets.max().item() + 1
    counts = torch.bincount(targets, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights

def evaluate_on_test(model, dataloader):
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    device = get_device()
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))


def main():
    dataloaders = get_dataloaders()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    class_names = dataloaders['train'].dataset.classes

    print(f"Classes: {class_names}")
    print(f"Dataset Sizes: {dataset_sizes}")
    

    # Use enhanced model
    model = CNN_GAP_Model()

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights_from_dataset(dataloaders['train'].dataset).to(DEVICE)
    print(f"Class weights: {class_weights.cpu().numpy().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )

    # ---- ADDED: Save final trained model ----
    save_model(model, os.path.join(MODELS_DIR, 'final_model_v2.pth'))
    # -----------------------------------------

    # Evaluate on test set
    evaluate_on_test(model, dataloaders['test'])


if __name__ == '__main__':
    main()
