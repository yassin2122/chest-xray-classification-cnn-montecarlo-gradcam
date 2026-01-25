import os
import torch

# Paths
# src/config.py -> src/ -> NN_Project/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = r"D:\seamster 5\my projects\Neural network\data\chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
IMSIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_CLASSES = 2
DROPOUT_P = 0.5
PATIENCE = 7

# Uncertainty / rejection defaults
# If the max class probability is below this, report "UNKNOWN"
CONFIDENCE_THRESHOLD = 0.80
# If predictive variance for the chosen class exceeds this, report "UNKNOWN"
# Note: scale depends on MC Dropout setup; tune with validation
UNCERTAINTY_THRESHOLD = 0.02
# Default number of MC Dropout samples for inference
MC_SAMPLES_DEFAULT = 30

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
SEED = 42
