import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version (from PyTorch):", torch.version.cuda)
    print("GPU detected:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected.")

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
