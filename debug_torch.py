import sys
print(f"Python version: {sys.version}")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {torch.device('cpu')}")
except Exception as e:
    print(f"Error importing torch: {e}")
