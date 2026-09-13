"""
src/config.py – Central configuration for paths, hyper-parameters, and device.

All paths are resolved relative to the repository root so the package works
regardless of where it is installed.  Override DATA_DIR via the environment
variable CHEST_XRAY_DATA_DIR when running on a different machine.
"""
import os
import torch

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file: src/config.py → src/ → root)
# ---------------------------------------------------------------------------
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(_SRC_DIR)          # repository root

# ---------------------------------------------------------------------------
# Data directories – override with env var on different machines
# ---------------------------------------------------------------------------
DATA_DIR  = os.environ.get(
    "CHEST_XRAY_DATA_DIR",
    os.path.join(BASE_DIR, "data", "chest_xray"),  # default relative path
)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# ---------------------------------------------------------------------------
# Output directories (created on first use)
# ---------------------------------------------------------------------------
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
IMSIZE        = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 25
LEARNING_RATE = 1e-3
NUM_CLASSES   = 2
DROPOUT_P     = 0.5
PATIENCE      = 7

# ---------------------------------------------------------------------------
# Uncertainty / rejection thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD  = 0.80   # predictions below this → "UNKNOWN"
UNCERTAINTY_THRESHOLD = 0.02   # predictive variance above this → "UNKNOWN"
MC_SAMPLES_DEFAULT    = 30     # default Monte Carlo Dropout forward passes

# ---------------------------------------------------------------------------
# Device & seed
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
