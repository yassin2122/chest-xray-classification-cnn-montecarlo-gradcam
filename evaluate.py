import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_dataloaders
from utils import get_device
from model import CNN_MLP_Model
from config import NUM_CLASSES, MODELS_DIR



def evaluate_model(model, dataloader, device):
    """
    Run evaluation and return predictions, true labels, and metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix_heatmap(y_true, y_pred, classes, save_path=None):
    """
    Plot and save confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    # Only import here to avoid circular dependencies if any
  
    device = get_device()
    model_path = os.path.join(MODELS_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print("Model not found. Please train first.")
        return

    model = CNN_MLP_Model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    dataloaders = get_dataloaders()
    test_loader = dataloaders['test']
    
    print("Evaluating on Test Set...")
    preds, labels = evaluate_model(model, test_loader, device)
    
    classes = test_loader.dataset.classes
    print(classification_report(labels, preds, target_names=classes))
    
    plot_path = os.path.join(os.path.dirname(MODELS_DIR), 'results', 'confusion', 'cm.png')
    plot_confusion_matrix_heatmap(labels, preds, classes, save_path=plot_path)

if __name__ == "__main__":
    main()
