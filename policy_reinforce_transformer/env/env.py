import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from itertools import combinations
import torch
from utils import get_device
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class TSPEnv:
    def __init__(self, batch_size=1 ,n_cities=25, coord_dim=2, fixed_coords=None):
        """
        TSP環境の初期化
        """
        self.batch_size = batch_size  # バッチサイズ
        self.n_cities = n_cities  # 都市の数
        self.coord_dim = coord_dim  # 座標の次元（例: x, y）
        self.data_coord_max = 1  # 座標の最大値
        self.fixed_coords = fixed_coords  # 固定された都市の座標
        self.device = get_device()
    
    def _generate_coords(self):
        """
        都市の座標を生成
        """
        if self.fixed_coords is not None:
            self.coords = self.fixed_coords
        else:
            coords = np.random.rand(self.batch_size, self.n_cities, self.coord_dim) * self.data_coord_max
            self.coords = coords.squeeze()
            # print("生成された都市の座標:")
            # print(self.coords)
        
    def reset(self):
        """
        環境のリセット
        - 訪問履歴、現在地、未訪問都市の初期化
        """
        # 座標を生成
        self._generate_coords()
        # print(f'coords: {self.coords}')
        self.visited_cities = np.zeros(self.n_cities)  # 訪問済み都市
        # print(f'visited_cities: {self.visited_cities}')
        self.total_distance = 0  # 総移動距離
        self.current_city = -1  # 現在の都市
        self.done = False
        self.step_counte = 1
        
        # state = self._get_state()
        data = self._get_data()
        return data, self.visited_cities

    def _get_data(self):
        if isinstance(self.coords, np.ndarray):
            self.coords = torch.tensor(self.coords, dtype=torch.float32)
        # すべての都市ペアを生成（i ≠ j）
        edges = list(combinations(range(self.n_cities), 2))
        # リスト → Tensor に変換し、転置して (2, N) の形にする
        edge_index = torch.tensor(edges, dtype=torch.long).T  # shape: (2, num_edges)

        # 双方向エッジを生成
        edge_index = to_undirected(edge_index)
        # print(f'edge_index: {edge_index}')

        # エッジの始点と終点の座標を取り出す
        src = edge_index[0]
        dst = edge_index[1]

        # ユークリッド距離（重み）を計算
        edge_weight = torch.norm(self.coords[src] - self.coords[dst], dim=1)
        # print(f'edge_weight: {edge_weight}')
        
        # `torch_geometric.data.Data` オブジェクト作成
        data = Data(x=self.coords.to(self.device), edge_index=edge_index.to(self.device), edge_weight=edge_weight.to(self.device))
        
        return data


    def step(self, action):
        """
        一巡する都市の並びを指定してステップを実行
        :param action: 一巡する都市のインデックス
        :return: next_state, reward, done
        """

        # 報酬の計算（現在地→次の都市の距離）
        reward = self._get_distance(action)

        self.visited_cities[action] = self.step_counte  # 訪問済み都市に追加
        self.step_counte += 1

        # print('env.py step')
        # print(f'visited_cities: {self.visited_cities}')
        
        self.current_city = action
        # print(f'current_city: {self.current_city}')
        
        # if np.sum(self.visited_cities) == self.n_cities:
        if np.all(self.visited_cities > 0):
            # print('All cities visited')
            # 最後の都市と最初の都市の距離を追加
            first_city = np.where(self.visited_cities == 1)[0][0]
            # print(f'first_city: {first_city}')
            reward += self._get_distance(first_city)

            # エピソード終了判定
            self.done = True
        
        # self.total_distance += reward
        # print(f'total_reward_env: {self.total_distance}')
        next_visited_cities = self.visited_cities.copy()  # 次の状態は訪問済み都市のコピー

        return next_visited_cities, reward, self.done


    def _get_distance(self, action):
        """
        訪問順序に基づく総移動距離を計算
        :param action: 訪問順序（例: [0, 3, 4, 1, 2]）
        :return: 総移動距離
        """
        # 現在の都市とactionの都市の距離を取得する
        if self.current_city == -1:
            distance = 0.0
        else:
            distance = -1 * np.linalg.norm(self.coords[self.current_city] - self.coords[action])
        return distance


# 動作確認   
def main():
    print('env.py main')
    env = TSPEnv()
    env.reset()
    action_list = [1, 4, 0, 2, 3]

    for i in action_list:
        action = i
        print(f'Action: {action}')
        next_visited_cities, reward, done = env.step(action)
        print(f'next_visited_cities: {next_visited_cities}')
        print(f'reward: {reward}')
        print(f'done: {done}')


if __name__ == '__main__':
    main()