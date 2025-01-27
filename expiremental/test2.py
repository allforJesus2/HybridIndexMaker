import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")