import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import Policy
import torch
from torch import optim
from utils import get_device
import numpy as np

class Agent:
    def __init__(self):
        self.gamma = 0.99  # 割引率
        self.lr = 0.001  # 学習率
        self.memory = []
        self.device = get_device()
        self.pi = Policy().to(self.device)  # ポリシーモデルの初期化
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0).to(self.device)  # shape: (1, 5, 4)
        action_probs = self.pi(state)[0]  # 都市の訪問確率を取得

        # 都市の訪問確率から次の都市を選択
        action = np.random.choice(len(action_probs), p=action_probs.detach().cpu().numpy())
        
        return action, action_probs
    
    def add(self, reward, action_probs):
        self.memory.append((reward, action_probs))

    def update(self):
        G, loss = 0, 0

        for reward, action_probs in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -torch.log(action_probs) * G
        
        loss = loss.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
    
    # modelを保存する関数
    def save_model(self, path):
        torch.save(self.pi.state_dict(), path)
    
    # modelをロードする関数
    def load_model(self, path):
        self.pi.load_state_dict(torch.load(path, weights_only=True))
    

def main():
    agent = Agent()
    state = np.array([
    [95.02370691, 54.86014805, 1.0, 161.25799003],
    [79.64110929, 9.31860416, 1.0, 161.25799003],
    [37.19301454, 14.77507159, 1.0, 161.25799003],
    [62.41494659, 80.49261453, 1.0, 161.25799003],
    [28.36552228, 90.45835087, 0.0, 161.25799003]
    ])

    agent.get_action(state)

if __name__ == '__main__':
    main()