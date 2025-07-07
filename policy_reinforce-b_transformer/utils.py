import torch
import json


def get_device():    
    # デバイスの定義 (GPUが利用可能であればGPUを使う)
    if torch.backends.mps.is_available():  # M1/M2チップのMetal API（GPU）対応
        device = torch.device("mps")
    elif torch.cuda.is_available():  # CUDA対応（通常のGPU）
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")  # CPU
    return device


def save_config(lr, gamma, save_path):
    """
    設定をJSON形式で保存する関数
    :param lr: 学習率
    :param gamma: 割引率
    :param save_path: モデルの保存パス（.pthファイル）
    """
    config = {"lr": lr, "gamma": gamma,}
    json_path = save_path.replace(".pth", ".json")
    with open(json_path, "w") as f:
        json.dump(config, f)



