import torch

print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")