import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import combinations
from model.model_transformer import Encoder
from model.model_transformer import Decoder
import torch
from torch import optim
from utils import get_device
from torch_geometric.data import Data
import numpy as np


class Agent:
    def __init__(self, lr=1e-4, gamma=0.99):
        self.gamma = gamma  # 割引率
        self.lr = lr  # 学習率
        self.memory = []
        self.device = get_device()
        self.encoder = Encoder().to(self.device)  # ポリシーモデルの初期化
        self.decoder = Decoder().to(self.device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr)
    
    def encoder_forward(self, data):
        # Encoder
        self.node_embeddings, self.graph_embed = self.encoder(data)  # GNNの出力を取得
        # print("node_embeddings.shape:", self.node_embeddings.shape)
        # print("graph_embed.shape:", self.graph_embed.shape)

    def get_action(self, visited_cities):
        visited_mask = torch.tensor(visited_cities, dtype=torch.bool, device=self.device).unsqueeze(0)  # shape: (1, 5)
        # print("visited_mask.shape:", visited_mask.shape)
        # print("visited_mask:", visited_mask)

        if not visited_mask.any():
            # 全て False（＝どの都市もまだ訪問していない）
            h_last = torch.empty(0, device=self.device)
            h_first = torch.empty(0, device=self.device)
            t = 0
        else:
            first_index = np.where(visited_cities == 1)[0][0]
            last_index = np.argmax(visited_cities)
            # print(f"first_index: {first_index}, last_index: {last_index}")
            h_first = self.node_embeddings[0, first_index, :].unsqueeze(0)  # shape: (1, 128)
            h_last = self.node_embeddings[0, last_index, :].unsqueeze(0)  # shape: (1, 128)
            t = visited_mask.sum().item()  # 訪問済み都市の数
        
        probs, _ = self.decoder(self.node_embeddings, self.graph_embed, h_last, h_first, visited_mask, t)
        # print("probs:", probs)
        # ノードを確率的に選ぶ（探索的）
        probs = probs.squeeze(0)  # shape: [10] に変換
        action = torch.multinomial(probs, num_samples=1).item()  # 0〜9の整数
        return action, probs[action]
    
    def add(self, reward, action_probs):
        self.memory.append((reward, action_probs))

    def update(self, random_baseline):
        G, loss = 0, 0
        EPS = 1e-8  # 極小値を定義

        for reward, action_probs in reversed(self.memory):
            # print(f"action_probs: {action_probs}")
            G = reward + self.gamma * G
            advantage = G - random_baseline  # ← ここがベースラインを引く部分！
            loss += -torch.log(action_probs + EPS) * advantage
        
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
        # 保存
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    # modelをロードする関数
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.encoder.to(self.device)
        self.decoder.to(self.device)    

def main():
    agent = Agent()

    device = get_device()

    state = torch.tensor([
        [95.02370691, 54.86014805],
        [79.64110929, 9.31860416],
        [37.19301454, 14.77507159],
        [62.41494659, 80.49261453],
        [28.36552228, 90.45835087]
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
    
    visited_cities = np.array([0, 2, 1, 0, 3])  # 各都市の訪問ステップ
    agent.encoder_forward(data)
    agent.get_action(visited_cities)
    # nex_node, probs = agent.get_action(data, state)
    # print('next_node:', nex_node)
    # print('probs:', probs)

if __name__ == '__main__':
    main()