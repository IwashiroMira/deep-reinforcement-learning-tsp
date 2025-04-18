import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import combinations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
from utils import get_device
import numpy as np


# LSTMを用いたTSPの順序モデル
class PolicyGCNLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=4, num_layers=2, output_size=1):
        super().__init__()
        self.device = get_device()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        # self.conv1 = GATConv(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout1 = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=0)

    def encode(self, x, edge_index, edge_weight):
        ''' グラフの特徴量をエンコードする '''
        x = self.conv1(x, edge_index, edge_weight)
        # x = self.relu(x)
        # x = self.conv2(x, edge_index, edge_weight)
        # x = self.relu(x)
        return x
    
    def decode(self, input):
        ''' エンコードされた特徴量をデコードする '''
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(input, (h0, c0))
        final_hidden = out[:, -1, :]  # [1, hidden_size] 最後の出力（文脈ベクトル）
        return final_hidden
    
    def forward(self, data, visited):
        ''' 順伝播 '''
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        z_all = self.encode(x, edge_index, edge_weight)
        
        # その順序通りに埋め込みを取得（系列データになる）
        z_visited_seq = z_all[visited]  # shape: [3, hidden_size]

        # LSTMは (batch, seq_len, input_size) を期待するので、batch 次元を追加
        lstm_input = z_visited_seq.unsqueeze(0)  # shape: [1, 3, hidden_size]

        final_hidden = self.decode(lstm_input)
        # scores = torch.matmul(z_all, final_hidden.squeeze(0).mT)  # [10]
        final_hidden_vec = final_hidden.squeeze(0).unsqueeze(-1)  # shape: [hidden_size, 1]
        scores = torch.matmul(z_all, final_hidden_vec).squeeze()  # shape: [num_nodes]
        scores[visited] = -float('inf')  # すでに訪問したノードは選ばれないように
        probs = torch.softmax(scores, dim=0)  # [10]
        
        # next_node = torch.argmax(probs).item()  # ← これが次に訪問するノード！
        next_node = torch.multinomial(probs, num_samples=1).item()

        return [next_node, probs]


def main():
    device = get_device()

    # モデルの初期化とデバイスへの移動
    model = PolicyGCNLSTM().to(device)

    state = torch.tensor([
        [95.02370691, 54.86014805, 1],
        [79.64110929, 9.31860416, 1],
        [37.19301454, 14.77507159, 0],
        [62.41494659, 80.49261453, 0],
        [28.36552228, 90.45835087, 0]
    ], dtype=torch.float32)

    # 都市数
    num_nodes = state.shape[0]

    # すべての都市ペアを生成（i ≠ j）
    edges = list(combinations(range(num_nodes), 2))

    # edge_index の作成
    edge_index = torch.tensor(edges, dtype=torch.long).T  # 転置して (2, N) の形にする

    # ユークリッド距離を計算してエッジの重みとする
    edge_weight = torch.norm(state[edge_index[0]] - state[edge_index[1]], dim=1)

    # Min-Maxスケーリングを追加 (0~1に正規化)
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())

    # 結果の出力
    print("edge_index:\n", edge_index)
    print("edge_weight:\n", edge_weight)

    # `torch_geometric.data.Data` オブジェクト作成
    data = Data(x=state.to(device), edge_index=edge_index.to(device), edge_weight=edge_weight.to(device))

    visited = [0, 1]
    # モデルの計算
    output = model(data, visited)

    print("訪問確率:", output[1])
    print("次に訪問するノード:", output[0])  # 出力形状: [32, 10]
if __name__ == '__main__':
    main()