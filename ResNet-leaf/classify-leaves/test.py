import torch

if torch.cuda.is_available():
    print("有可用的 GPU")
else:
    print("没有可用的 GPU")
