import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import combinations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from utils import get_device
import numpy as np


# Multi-Head Attentionを用いたTSPのモデル
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim=128, ff_hidden_dim=512):
        super().__init__()
        self.device = get_device()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, hidden_dim)
        )
    
    def forward(self, h):
        # Multi-head attention + skip + BN
        h_attn, _ = self.mha(h, h, h)
        # print("h_attn.shape:", h_attn.shape)
        h = self.norm1((h + h_attn).squeeze(0)).unsqueeze(0)  # shape: (1, 5, 128)

        # Feed Forward Network
        h_ff = self.ff(h)
        h = self.norm2((h + h_ff).squeeze(0)).unsqueeze(0)
        return h  # N, D

class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, ff_hidden_dim=512, n_layers=3):
        super().__init__()
        self.device = get_device()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, ff_hidden_dim) for _ in range(n_layers)
        ])
        
    def forward(self, data):
        ''' 順伝播 '''
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        h = self.conv1(x, edge_index, edge_weight)
        h = h.unsqueeze(0)  # shape: (1, 5, 128)
        # print("GNN h.shape:", h.shape)

        for layer in self.layers:
            h = layer(h)
        
        graph_embed = h.mean(dim=1)
        # print("last h.shape:", h.shape)
        # print("graph_embed.shape:", graph_embed.shape)
        return h, graph_embed

class Decoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, tanh_clipping=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping

        # 初期パラメータ
        self.v_l = nn.Parameter(torch.randn(1, embed_dim))
        self.v_f = nn.Parameter(torch.randn(1, embed_dim))

        # 線形変換層
        self.w_q = nn.Linear(3 * embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def build_context(self, graph_embed, h_last, h_first, t):
        if t == 0:
            context = torch.cat([
                graph_embed,
                self.v_l.expand_as(graph_embed),
                self.v_f.expand_as(graph_embed)
            ], dim=-1)
        else:
            context = torch.cat([graph_embed, h_last, h_first], dim=-1)
        return context
    
    def forward(self, node_embedding, graph_embed, h_last, h_first, visited_mask, t):
        # ノードの埋め込みを取得
        # print("node_embedding.shape:", node_embedding.shape)
        B, N, D = node_embedding.shape  # B: バッチサイズ, N: ノード数, D: 埋め込み次元

        # 1. コンテキストベクトルとQuery
        context = self.build_context(graph_embed, h_last, h_first, t)
        q = self.w_q(context).unsqueeze(0)  # shape: (1, 128)
        # print("q.shape:", q.shape)

        # 2. KeyとValue
        k = self.w_k(node_embedding)  # shape: (B, N, D)
        v = self.w_v(node_embedding)  # shape: (B, N, D)
        # print("k.shape:", k.shape)
        # print("v.shape:", v.shape)

        # 3. Attentionの計算
        attn_output, attn_weights = self.mha(q, k, v, key_padding_mask=visited_mask)  # shape: (1, B, D)
        # print("attn_output.shape:", attn_output.shape)
        h_c_new = attn_output.squeeze(1)  # shape: (B, D)
        # print("h_c_new.shape:", h_c_new.shape)

        # 4. logitsの計算
        q_logits = h_c_new
        k_logits = k
        logits = torch.matmul(q_logits.unsqueeze(1), k_logits.transpose(1, 2)).squeeze(1) / (D ** 0.5)  # shape: (B, N)
        # print("logits.shape:", logits.shape)

        # 5. クリッピングとマスク
        logits = self.tanh_clipping * torch.tanh(logits)
        logits = logits.masked_fill(visited_mask, float('-inf'))  # 訪問済み都市をマスク

        # 6. softmax
        probs = torch.softmax(logits, dim=-1)  # shape: (B, N)
        # print("probs.shape:", probs.shape)

        return probs, logits

def main():
    device = get_device()

    # モデルの初期化とデバイスへの移動
    encoder = Encoder().to(device)

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

    # 結果の出力
    # print("edge_index:\n", edge_index)
    # print("edge_weight:\n", edge_weight)

    # `torch_geometric.data.Data` オブジェクト作成
    data = Data(x=state.to(device), edge_index=edge_index.to(device), edge_weight=edge_weight.to(device))

    # モデルの計算
    node_embeddings, graph_embed = encoder(data)

    # print("node_embeddings.shape:", node_embeddings.shape)
    # print("graph_embed.shape:", graph_embed.shape)

    # ''' Decoder '''
    # decoder = Decoder().to(device)
    # # t=0の時
    # h_last = torch.empty(0, device=device)
    # h_first = torch.empty(0, device=device)
    # visited_mask = torch.zeros((1, num_nodes), dtype=torch.bool, device=device)
    # print("visited_mask:", visited_mask)
    # probs, logits = decoder(node_embeddings, graph_embed, h_last, h_first, visited_mask, 0)
    # print(f"probs:", probs)

if __name__ == '__main__':
    main()