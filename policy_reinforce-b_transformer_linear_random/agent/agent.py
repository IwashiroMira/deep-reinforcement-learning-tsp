import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_transformer import PolicyNet
import torch
from utils import get_device
from env.env import TSPEnv
import numpy as np


class Agent:
    def __init__(self, lr=1e-5, gamma=0.99):
        self.gamma = gamma  # 割引率
        self.lr = lr  # 学習率
        self.memory = []
        self.device = get_device()
        self.model = PolicyNet().to(self.device)  # ポリシーモデルの初期化
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def encoder_forward(self, data):
        # Encoder
        self.model.encoder_forward(data)  # GNNの出力を取得
        # print("node_embeddings.shape:", self.node_embeddings.shape)
        # print("graph_embed.shape:", self.graph_embed.shape)

    def get_action(self, visited_cities):
        probs = self.model.decoder_forward(visited_cities)  # デコーダーのフォワードパスを実行
        # print("probs.shape:", probs.shape)
        # print("probs:", probs)
        # ノードを確率的に選ぶ（探索的）
        # バッチ対応版（actionは shape: (B,) のテンソル）
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)  # shape: (B,)
        # print("actions.shape:", actions.shape)
        # print("actions:", actions)
        selected_probs = probs[torch.arange(probs.size(0)), actions]
        # print(selected_probs) 
        return actions, selected_probs
    
    def add(self, reward, action_probs):
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)  # ← ここ重要
        self.memory.append((reward, action_probs))
        # print(f"memory: {self.memory}")
    
    def update(self, bl_reward=None):
        G, loss = 0, 0
        if bl_reward is not None:
            bl_reward = torch.tensor(bl_reward, dtype=torch.float32, device=self.device)

        for reward, action_probs in reversed(self.memory):
            # print(f"action_probs: {action_probs}")
            G = reward + self.gamma * G
            advantage = G if bl_reward is None else (G - bl_reward)
            # print(f"G: {G}")
            loss += -torch.log(action_probs) * advantage
            # print(f"loss: {loss}")
        
        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        self.memory = []
    
    # def update(self):
    #     G, loss = 0, 0
    #     EPS = 1e-8  # 極小値を定義

    #     for reward, action_probs in reversed(self.memory):
    #         # print(f"action_probs: {action_probs}")
    #         G = reward + self.gamma * G
    #         # print(f"G: {G}")
    #         loss += -torch.log(action_probs) * G  # EPSがないとlog(0)でエラーになる
    #         # print(f"loss: {loss}")
        
    #     loss = loss.sum()
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
            
    #     self.memory = []
    
    # modelを保存する関数
    def save_model(self, path):
        # 保存
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    # modelをロードする関数
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.to(self.device)

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