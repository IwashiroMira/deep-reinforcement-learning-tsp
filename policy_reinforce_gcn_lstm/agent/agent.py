import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import combinations
from model.model_gcn_lstm import PolicyGCNLSTM as Policy
import torch
from torch import optim
from utils import get_device
from torch_geometric.data import Data
import numpy as np

class Agent:
    def __init__(self, lr=1e-4, gamma=0.99, hidden_size=4):
        self.gamma = gamma  # 割引率
        self.lr = lr  # 学習率
        self.memory = []
        self.device = get_device()
        self.pi = Policy(hidden_size=hidden_size).to(self.device)  # ポリシーモデルの初期化
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, data, state):
        
        # print('state:')
        # print(state)

        # すでに tensor なのでそのまま使って OK！
        state = state.unsqueeze(0).to(self.device)  # shape: (1, 5, 3)

        # 訪問済み都市のフラグを取り出す（3列目）
        visited_cities = state[:, :, 3].squeeze()   # shape: (5,)
        # visited: tensor([0., 1., 0., 0., 0.], dtype=torch.float32)
        visited_indices = (visited_cities == 1).nonzero(as_tuple=True)[0]  # ← LongTensorになる！

        # ログ（確認用）
        # print(f'visited_cities: {visited_cities}')
        # print(f'visited_indices: {visited_indices}')
        action, probs = self.pi(data, visited_indices)  # 都市の訪問確率を取得
        # print('action_logits:')
        # print(action_logits)
 
        return action, probs
    
    def add(self, reward, action_probs):
        self.memory.append((reward, action_probs))

    def update(self):
        G, loss = 0, 0
        EPS = 1e-8  # 極小値を定義

        for reward, action_probs in reversed(self.memory):
            # print(f"action_probs: {action_probs}")
            G = reward + self.gamma * G
            loss += -torch.log(action_probs + EPS) * G  # EPSがないとlog(0)でエラーになる
        
        loss = loss.sum()
        
        self.optimizer.zero_grad()
        # **学習前の重みを表示**
        # print("🔍 LSTM weights BEFORE update:")
        # for name, param in self.pi.lstm.named_parameters():
        #     print(f"{name}: {param.data}")
            
        loss.backward()
        self.optimizer.step()

        # **学習後の重みを表示**
        # print("🔍 LSTM weights AFTER update:")
        # for name, param in self.pi.lstm.named_parameters():
        #     print(f"{name}: {param.data}")
            
        self.memory = []
    
    # modelを保存する関数
    def save_model(self, path):
        torch.save(self.pi.state_dict(), path)
    
    # modelをロードする関数
    def load_model(self, path):
        self.pi.load_state_dict(torch.load(path, weights_only=True))
    

def main():
    agent = Agent()

    device = get_device()

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

    # `torch_geometric.data.Data` オブジェクト作成
    data = Data(x=state.to(device), edge_index=edge_index.to(device), edge_weight=edge_weight.to(device))
    print('data:')
    print(data, state)
    
    nex_node, probs = agent.get_action(data, state)
    print('next_node:', nex_node)
    print('probs:', probs)

if __name__ == '__main__':
    main()