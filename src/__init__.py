"""
Chest X-Ray Classification – src package.

Public API surface:
    from src.config    import DEVICE, NUM_CLASSES, ...
    from src.model     import CNN_GAP_Model, CNN_MLP_Model
    from src.dataset   import get_dataloaders
    from src.transforms import data_transforms
    from src.utils     import get_device, save_model, seed_everything
    from src.explain   import GradCAM
"""

__version__ = "0.1.0"
__all__ = [
    "config",
    "model",
    "dataset",
    "transforms",
    "utils",
    "explain",
    "train",
    "evaluate",
]
