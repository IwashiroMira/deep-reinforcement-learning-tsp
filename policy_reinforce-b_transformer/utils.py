import torch

def get_device():    
    # デバイスの定義 (GPUが利用可能であればGPUを使う)
    if torch.backends.mps.is_available():  # M1/M2チップのMetal API（GPU）対応
        device = torch.device("mps")
    elif torch.cuda.is_available():  # CUDA対応（通常のGPU）
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")  # CPU
    return device

