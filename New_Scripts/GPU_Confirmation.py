import os
import torch

# Check if PyTorch can access GPUs
if torch.cuda.is_available():
    print("PyTorch is using GPU.")
    print("Number of GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("PyTorch is using CPU.")

# Ensure environment is set properly
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Verify PyTorch and CUDA versions
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "CUDA not available")
