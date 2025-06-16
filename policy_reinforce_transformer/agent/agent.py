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
from env.env import TSPEnv
from torch_geometric.data import Data
import numpy as np


class Agent:
    def __init__(self, lr=1e-5, gamma=0.99):
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
        B, N, D = self.node_embeddings.shape  # B: バッチサイズ, N: 都市の数, D: 埋め込み次元
        # print("visited_cities.shape:", visited_cities.shape)
        visited_mask = torch.tensor(visited_cities, dtype=torch.bool, device=self.device)  # shape: (1, 5)
        # print("visited_mask.shape:", visited_mask.shape)
        # print("visited_mask:", visited_mask)

        if not visited_mask.any():
            # 全て False（＝どの都市もまだ訪問していない）
            h_last = torch.empty(0, device=self.device)
            h_first = torch.empty(0, device=self.device)
            t = 0
        else:
            first_index = np.where(visited_cities == 1)[0][0]
            first_index = np.array([
                np.where(row == 1)[0][0] for row in visited_cities
            ])
            last_index = np.argmax(visited_cities,axis=1)
            # print(f"first_index: {first_index}, last_index: {last_index}")
            
            h_first = self.node_embeddings[torch.arange(B), first_index, :]  # shape: (B, D)
            h_last  = self.node_embeddings[torch.arange(B), last_index, :]   # shape: (B, D)

            # print("agent h_first.shape:", h_first.shape)
            # print("agent h_last.shape:", h_last.shape)
            t = visited_mask.sum().item()  # 訪問済み都市の数
        
        probs, _ = self.decoder(self.node_embeddings, self.graph_embed, h_last, h_first, visited_mask, t)
        # print("probs.shape:", probs.shape)
        # print("probs:", probs)
        # ノードを確率的に選ぶ（探索的）
        # バッチ対応版（actionは shape: (B,) のテンソル）
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)  # shape: (B,)
        # print("actions.shape:", actions.shape)
        # print("actions:", actions)
        selected_probs = probs[torch.arange(B), actions]
        # print(selected_probs) 

        return actions, selected_probs
    
    def add(self, reward, action_probs):
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)  # ← ここ重要
        self.memory.append((reward, action_probs))
        # print(f"memory: {self.memory}")

    def update(self):
        G, loss = 0, 0
        EPS = 1e-8  # 極小値を定義

        for reward, action_probs in reversed(self.memory):
            # print(f"action_probs: {action_probs}")
            G = reward + self.gamma * G
            # print(f"G: {G}")
            loss += -torch.log(action_probs + EPS) * G  # EPSがないとlog(0)でエラーになる
            # print(f"loss: {loss}")
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
    env = TSPEnv(batch_size=2, n_cities=5)
    agent = Agent()
    data, visited_cities = env.reset()
    # print(f'visited_cities: {visited_cities}')
    # agent.encoder_forward(data)
    
    visited_cities_zero = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]])  # 各都市の訪問ステップ
    agent.encoder_forward(data)
    action, probs = agent.get_action(visited_cities_zero)
    print('action:', action)
    print('probs:', probs)

    visited_cities = np.array([[0, 2, 1, 0, 3],
                               [1, 2, 3, 0, 0]])  # 各都市の訪問ステップ
    agent.encoder_forward(data)
    action, probs = agent.get_action(visited_cities)
    print('action:', action)
    print('probs:', probs)

if __name__ == '__main__':
    main()