import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from config import NUM_CLASSES, DROPOUT_P
# -----------------------------
# CNN + MLP Model for Chest X-ray Classification
# -----------------------------
class CNN_MLP_Model(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_p=DROPOUT_P):
        super().__init__()

        # -------- CNN Feature Extractor --------
        # Conv layer 1: 3 input channels -> 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Conv layer 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Max pooling to reduce spatial size
        self.pool = nn.MaxPool2d(2, 2)  # halves height & width

        # -------- MLP Classifier --------
        # Fully connected layers after flattening
        self.fc1 = nn.Linear(64 * 56 * 56, 256)  # input size depends on CNN output
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Flatten for MLP
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop2(x)

        # Output logits
        return self.fc3(x)

    # Monte Carlo Dropout for uncertainty estimation
    def mc_forward(self, x, n_samples=30):
        self.train()  # keep dropout active
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                preds.append(probs.unsqueeze(0))

        preds = torch.cat(preds, dim=0)
        mean_probs = preds.mean(dim=0)
        uncertainty = preds.var(dim=0)

        return mean_probs, uncertainty

def get_model_summary(model, input_size=(3, 224, 224)):
    """
    Prints model summary and parameter counts.  
    """
    summary(model, input_size)  # shows layer output sizes and total params

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

# -----------------------------
# Enhanced CNN with Global Average Pooling (fewer params, better generalization)
# -----------------------------
class CNN_GAP_Model(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_p=DROPOUT_P):
        super().__init__()

        # Feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)

        # Global average pooling to drastically reduce parameters
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.gap(x)
        return self.classifier(x)