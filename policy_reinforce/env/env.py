import numpy as np

class TSPEnv:
    def __init__(self, batch_size=1 ,n_cities=5, coord_dim=2, fixed_coords=None):
        """
        TSP環境の初期化
        """
        self.batch_size = batch_size  # バッチサイズ
        self.n_cities = n_cities  # 都市の数
        self.coord_dim = coord_dim  # 座標の次元（例: x, y）
        self.data_coord_max = 100  # 座標の最大値
        self.fixed_coords = fixed_coords  # 固定された都市の座標

    
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
        self.visted_cities = np.zeros(self.n_cities)  # 訪問済み都市
        self.total_distance = 0  # 総移動距離
        self.current_city = 0  # 現在の都市
        self.done = False
        
        self.visted_cities[0] = 1  # 最初の都市は訪問済み
        state = self._get_state()
        return state

    def _get_state(self):
        """
        現在の状態を返す
        :return: 現在の状態
        """
        state = np.hstack([
            self.coords,  # 都市の座標
            self.visted_cities.reshape(-1, 1),  # 訪問済み都市
            np.full((self.n_cities, 1), self.total_distance)  # 現在の都市
        ])
        return state

    def step(self, action):
        """
        一巡する都市の並びを指定してステップを実行
        :param action: 一巡する都市のインデックス
        :return: next_state, reward, done
        """

        # 報酬の計算（現在地→次の都市の距離）
        reward = self._get_distance(action)

        self.visted_cities[action] += 1  # 訪問済み都市に追加
        
        self.current_city = action
        

        # if np.sum(self.visted_cities) == self.n_cities:
        if np.all(self.visted_cities > 0):
            # 最後の都市と最初の都市の距離を追加
            reward += self._get_distance(0)

            # エピソード終了判定
            self.done = True
        
        self.total_distance += reward
        next_state = self._get_state()

        return next_state, reward, self.done


    def _get_distance(self, action):
        """
        訪問順序に基づく総移動距離を計算
        :param action: 訪問順序（例: [0, 3, 4, 1, 2]）
        :return: 総移動距離
        """
        # 現在の都市とactionの都市の距離を取得する
        distance = np.linalg.norm(self.coords[self.current_city] - self.coords[action])
        return 1 * distance


# 動作確認   
def main():
    print('env.py main')
    env = TSPEnv()
    env.reset()
    print(env.state)

    for i in range(1,5):
        action = i
        print(f'Action: {action}')
        next_state, reward, done = env.step(action)
        print(f'next_state: {next_state}')
        print(f'reward: {reward}')
        print(f'done: {done}')


if __name__ == '__main__':
    main()