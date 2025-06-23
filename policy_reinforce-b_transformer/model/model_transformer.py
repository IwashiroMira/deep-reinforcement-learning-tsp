import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import combinations
from env.env import TSPEnv
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from utils import get_device
import numpy as np


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = get_device()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encoder_forward(self, data):
        # Encoder
        self.node_embeddings, self.graph_embed = self.encoder(data)  # GNNの出力を取得
        # print("node_embeddings.shape:", self.node_embeddings.shape)
        # print("graph_embed.shape:", self.graph_embed.shape)
    
    def visit_cities_to_mask(self, visited_cities):
        ''' 訪問済み都市をマスクに変換 '''
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

        return h_last, h_first, visited_mask, t
    
    def decoder_forward(self, visited_cities):
        h_last, h_first, visited_mask, t = self.visit_cities_to_mask(visited_cities)        
        probs, _ = self.decoder(self.node_embeddings, self.graph_embed, h_last, h_first, visited_mask, t)
        return probs  # shape: (B, N)


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
        B, N, F = h.shape
        # Multi-head attention + skip + BN
        h_attn, _ = self.mha(h, h, h)
        # print("h_attn.shape:", h_attn.shape)
        h_res = h + h_attn
        h_bn = h_res.view(B * N, F)
        h_bn = self.norm1(h_bn)
        h = h_bn.view(B, N, F)
        # print("norm1 h.shape:", h.shape)

        # Feed Forward Network
        h_ff = self.ff(h)
        # print("h_ff.shape:", h_ff.shape)
        h_res2 = h + h_ff

        h_bn2 = h_res2.view(B * N, F)
        h_bn2 = self.norm2(h_bn2)
        h = h_bn2.view(B, N, F)
        # print("norm1 h.shape:", h.shape)

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
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        h = self.conv1(x, edge_index, edge_weight)

        batch_size = data.num_graphs
        num_nodes = h.size(0) // batch_size
        h = h.view(batch_size, num_nodes, -1)  # → (B, N, F)
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
            B = graph_embed.shape[0]
            h_last = self.v_l.expand(B, -1)
            h_first = self.v_f.expand(B, -1)
        
        # print("h_last.shape:", h_last.shape)  # (B, D)
        # print("h_first.shape:", h_first.shape) # (B, D)
        
        context = torch.cat([graph_embed, h_last, h_first], dim=-1)
        return context
    
    def forward(self, node_embedding, graph_embed, h_last, h_first, visited_mask, t):
        # ノードの埋め込みを取得
        # print("node_embedding.shape:", node_embedding.shape)
        B, N, D = node_embedding.shape  # B: バッチサイズ, N: ノード数, D: 埋め込み次元

        # 1. コンテキストベクトルとQuery
        context = self.build_context(graph_embed, h_last, h_first, t)
        # print("context.shape:", context.shape)
        q = self.w_q(context).unsqueeze(1)  # shape: (1, 128)
        # print("q.shape:", q.shape)

        # 2. KeyとValue
        k = self.w_k(node_embedding)  # shape: (B, N, D)
        v = self.w_v(node_embedding)  # shape: (B, N, D)
        # print("k.shape:", k.shape)
        # print("v.shape:", v.shape)

        # 3. Attentionの計算
        # print(f"visited_mask: {visited_mask}")
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
        # print("after mask logits.shape:", logits.shape)

        # 6. softmax
        probs = torch.softmax(logits, dim=-1)  # shape: (B, N)
        # print("probs.shape:", probs.shape)

        return probs, logits

def main():
    device = get_device()
    env = TSPEnv()
    data, visited_cities = env.reset()
    print(visited_cities)

    # モデルの初期化とデバイスへの移動
    encoder = Encoder().to(device)
    # モデルの計算
    node_embeddings, graph_embed = encoder(data)

    # print("node_embeddings.shape:", node_embeddings.shape)
    # print("graph_embed.shape:", graph_embed.shape)
    
    batch_size, num_nodes, embed_dim = node_embeddings.shape

    ''' Decoder '''
    decoder = Decoder().to(device)
    # t=0の時
    h_last = torch.empty(0, device=device)
    h_first = torch.empty(0, device=device)
    visited_mask = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)
    # print("visited_mask:", visited_mask)
    probs, logits = decoder(node_embeddings, graph_embed, h_last, h_first, visited_mask, 0)
    # print(f"probs:", probs)

if __name__ == '__main__':
    main()