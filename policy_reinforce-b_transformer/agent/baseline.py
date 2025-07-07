import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_transformer import PolicyNet
import torch
from utils import get_device


class BaseLine():
    def __init__(self, lr, gamma):
        self.gamma = gamma  # 割引率
        self.lr = lr  # 学習率
        self.memory = []
        self.device = get_device()
        self.model = PolicyNet().to(self.device)  # ポリシーモデルの初期化
    
    def encoder_forward(self, data):
        # Encoder
        self.model.encoder_forward(data)  # GNNの出力を取得
        # print("node_embeddings.shape:", self.node_embeddings.shape)
        # print("graph_embed.shape:", self.graph_embed.shape)

    def get_action(self, visited_cities):
        self.model.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            # デコーダーのフォワードパスを実行
            probs = self.model.decoder_forward(visited_cities)  # デコーダーのフォワードパスを実行
            # print("probs.shape:", probs.shape)
            # print("probs:", probs)
            # ノードをに選ぶ（探索的）
            # バッチ対応版（actionは shape: (B,) のテンソル）
            actions = torch.argmax(probs, dim=1)  # shape: (B,)
            # print("actions.shape:", actions.shape)
            # print("actions:", actions)
            selected_probs = probs[torch.arange(probs.size(0)), actions]
            # print(selected_probs) 
        return actions, selected_probs
