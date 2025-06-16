import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import TSPEnv
from agent.agent import Agent
import json
import argparse
import sys


# input_data = np.array([
#     [40,57],
#     [51,28],
#     [57,  3],
#     [45, 94],
#     [58, 28],
# ])
input_data = np.array([
    [0.33333333, 0.84693878],
    [0.54166667, 0.        ],
    [0.19791667, 0.05102041],
    [0.02083333, 0.64285714],
    [0.59375,    0.56122449],
    [0.22916667, 0.90816327],
    [0.76041667, 0.84693878],
    [0.04166667, 0.95918367],
    [0.52083333, 0.98979592],
    [0.16666667, 0.86734694],
    [0.34375,    0.78571429],
    [0.32291667, 0.98979592],
    [0.09375,    0.23469388],
    [0.22916667, 0.7244898 ],
    [0.07291667, 1.        ],
    [0.30208333, 0.67346939],
    [1.,         0.90816327],
    [0.75,       0.60204082],
    [0.09375,    0.29591837],
    [0.5,        0.84693878],
    [0.08333333, 0.79591837],
    [0.82291667, 0.84693878],
    [0.,         0.76530612],
    [0.47916667, 0.37755102],
    [0.42708333, 0.87755102]
])
# input_data = np.array([
#     [34, 83],
#     [54,  0],
#     [21,  5],
#     [ 4, 63],
#     [59, 55],
#     [24, 89],
#     [75, 83],
#     [ 6, 94],
#     [52, 97],
#     [18, 85],
# ])

def plot_route(reward_history):
    # 最小トータルリワードのエピソードを取得
    best_episode = max(reward_history, key=lambda x: x["total_reward"])
    best_reward = best_episode["total_reward"]
    best_order = best_episode["visit_orders"]
    visit_order_list = [v.item() for v in best_order]
    print(f"visit_order_list: {visit_order_list}")
    print(f'best_reward: {best_reward}')

    # 訪問順序の最初と最後に 0 を追加して巡回経路にする
    best_order.append(best_order[0])
    
    # 訪問順序に基づく座標を取得
    ordered_coords = [input_data[i] for i in best_order]

    # プロットの作成
    plt.figure(figsize=(6, 6))
    plt.scatter(input_data[:, 0], input_data[:, 1], color='red', label='Cities')
    plt.plot(*zip(*ordered_coords), marker='o', linestyle='-', color='blue', label='Best Route')

    # 各都市にラベルを付与
    for i, (x, y) in enumerate(input_data):
        plt.text(x, y, str(i), fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Best Route with Total Reward: {-1 * best_reward}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Best Route")
    plt.close()  # 状態をリセット（重要）
    # plt.show()

def plot_reward_history(reward_history_rl, reward_history_random):
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history_rl, label='RL', linestyle='-', marker='o')
    plt.plot(reward_history_random, label='Random', linestyle='--', marker='x')
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward History')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Best Route Reward History")
    plt.close()  # 状態をリセット（重要）
    # plt.show()

def calc_distance_random_order(input_data):
    # 訪問順序をランダムに生成
    n_cities = len(input_data)
    visit_orders = np.random.permutation(n_cities)
    visit_orders = np.append(visit_orders, visit_orders[0])  # 最初の都市に戻る

    # 訪問順序に基づく座標を取得
    ordered_coords = [input_data[i] for i in visit_orders]

    # 総移動距離の計算
    total_distance = 0
    for i in range(n_cities):
        total_distance += np.linalg.norm(ordered_coords[i] - ordered_coords[i + 1])

    return total_distance


def main(model_path='save/model.pth', episodes=100, plot=True):
    # 同名の .json ファイルからパラメータ読み込み
    config_path = model_path.replace(".pth", ".json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # config を使ってAgent構築
    agent = Agent(
        lr=config["lr"],
        gamma=config["gamma"]
    )
    env = TSPEnv(batch_size=1, n_cities=25, fixed_coords=input_data)
    reward_history = []
    reward_history_random = []

    # 保存済みモデルをロード
    agent.load_model(model_path)

    for episode in range(episodes):
        # print('episode:', episode)
        data, visited_cities = env.reset()
        agent.encoder_forward(data)
        done = False
        total_reward = 0
        visit_orders = []

        while not done:
            action, probs = agent.get_action(visited_cities)
            visit_orders.append(action)
            next_visited_cities, reward, done = env.step(action)
            
            total_reward += reward
            visited_cities = next_visited_cities

        reward_history.append({
            "total_reward": total_reward,
            "visit_orders": visit_orders
            })

        # print(f'Episode: {episode}, Total Reward: {total_reward}')
        # print(f'visit_orders:{visit_orders}')

        # ランダムな訪問順序の総移動距離を計算
        total_distance_random = calc_distance_random_order(input_data)
        reward_history_random.append(total_distance_random)

        # print(f'Episode: {episode}, Total Reward Random: {total_distance_random}')
    
    # データの準備
    reward_history_rl = [-1 * r["total_reward"] for r in reward_history]
    reward_history_random = reward_history_random

    # Best Modelでグラフをプロット
    if plot:
        plot_route(reward_history)  # 最短経路のプロット
        plot_reward_history(reward_history_rl, reward_history_random)

    min_reward = min(reward_history_rl)  # 最小値
    # mean_reward = np.mean(reward_history_rl) 
    print(float(min_reward.item()))  # 文字列入れない、optunaでエラーになる


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs='?', default="save/model.pth")
    parser.add_argument("--plot", action="store_true", help="プロットを表示するかどうか")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    main(args.model_path, args.episodes, plot=args.plot)