import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from utils import get_device


# LSTMを用いたTSPの順序モデル
class Policy(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=5):
        super().__init__()
        self.device = get_device()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 最後のタイムステップの出力だけを使用
        out = torch.softmax(out, dim=1)  # ソフトマックス関数で確率に変換
        return out


def main():
    device = get_device()

    # モデルの初期化とデバイスへの移動
    model = Policy().to(device)

    # 都市数5、座標次元2
    state = torch.tensor([
    [95.02370691, 54.86014805, 1.0, 161.25799003],
    [79.64110929, 9.31860416, 1.0, 161.25799003],
    [37.19301454, 14.77507159, 1.0, 161.25799003],
    [62.41494659, 80.49261453, 1.0, 161.25799003],
    [28.36552228, 90.45835087, 0.0, 161.25799003]
    ], dtype=torch.float32)

    state = state.unsqueeze(0).to(device)  # shape: (1, 5, 4)

    # モデルの計算
    output = model(state)
    print(output)
    print(output.shape)  # 出力形状: [32, 10]

if __name__ == '__main__':
    main()