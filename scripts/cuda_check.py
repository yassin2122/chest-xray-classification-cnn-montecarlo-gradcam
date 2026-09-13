"""
cuda_check.py  –  Quick CUDA / PyTorch environment sanity-check.

Run from the project root:
    python scripts/cuda_check.py
"""
import torch


def main() -> None:
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"GPU detected    : {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected – running on CPU.")


if __name__ == "__main__":
    main()
