"""
tests/test_transforms.py – Smoke-tests for image pre-processing pipelines.
"""
import torch
from PIL import Image
import numpy as np
from src.transforms import data_transforms


def _make_pil_image(h: int = 256, w: int = 256) -> Image.Image:
    arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


class TestDataTransforms:
    def test_train_output_shape(self):
        img = _make_pil_image()
        tensor = data_transforms["train"](img)
        assert tensor.shape == (3, 224, 224)

    def test_val_output_shape(self):
        img = _make_pil_image()
        tensor = data_transforms["val"](img)
        assert tensor.shape == (3, 224, 224)

    def test_test_output_shape(self):
        img = _make_pil_image()
        tensor = data_transforms["test"](img)
        assert tensor.shape == (3, 224, 224)

    def test_output_is_float_tensor(self):
        img = _make_pil_image()
        tensor = data_transforms["test"](img)
        assert tensor.dtype == torch.float32
