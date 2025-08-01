import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import TSPEnv
from agent.agent import Agent
from agent.baseline import BaseLine
import copy
import argparse
import sys
from config import environment, inference
import utils

# n=48
# input_data = np.array([
#     [0.86738906, 0.27889447],
#     [0.28676471, 0.        ],
#     [0.7120743,  0.27328952],
#     [0.0504386,  0.16061075],
#     [0.39628483, 0.31580982],
#     [0.98013416, 0.85968303],
#     [0.9756192,  0.71627368],
#     [0.93588751, 0.24313877],
#     [0.88854489, 0.36238887],
#     [0.14215686, 0.39408581],
#     [0.70407637, 0.50173947],
#     [0.77128483, 0.55334364],
#     [0.60577915, 0.5148821 ],
#     [0.59365325, 0.39137998],
#     [0.81746646, 0.51662157],
#     [0.78650671, 0.12736761],
#     [0.98052116, 1.        ],
#     [0.96130031, 0.69192114],
#     [0.99613003, 0.91090066],
#     [0.75980392, 0.6863162 ],
#     [0.57701238, 0.64920758],
#     [0.78573271, 0.21260147],
#     [0.66937564, 0.41979126],
#     [0.20936533, 0.5409741 ],
#     [0.55430857, 0.44684963],
#     [0.08578431, 0.19250097],
#     [0.97329721, 0.92945497],
#     [0.97149123, 0.7674913 ],
#     [0.40853973, 0.14418245],
#     [0.94711042, 0.86896019],
#     [0.97200722, 0.53942791],
#     [0.41731166, 0.63683804],
#     [0.82765738, 0.61132586],
#     [0.59313725, 0.22960959],
#     [0.00167699, 0.42636258],
#     [0.93369453, 0.72844994],
#     [1.,         0.88616158],
#     [0.95227038, 0.43177426],
#     [0.44814241, 0.54483958],
#     [0.80766254, 0.41070738],
#     [0.64176987, 0.02512563],
#     [0.24587203, 0.30131426],
#     [0.9378225,  0.94491689],
#     [0.96736326, 0.62408195],
#     [0.,         0.51526865],
#     [0.87680599, 0.57653653],
#     [0.66756966, 0.62775416],
#     [0.38867389, 0.37340549],
# ])

# n=25
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

# B×N×D に reshape（B=1）
input_data = input_data[np.newaxis, :, :]  # shape = (1, 25, 2)

def main(model_path='save/model.pth', episodes=100, plot=True):
    # 同名の .json ファイルからパラメータ読み込み
    if os.path.exists(model_path):
        config = utils.load_config(model_path)

    # config を使ってAgentとbaseline構築
    agent = Agent(
        lr=config["lr"],
        gamma=config["gamma"]
    )
    baseline = BaseLine(
        lr=config["lr"],
        gamma=config["gamma"]
    )

    # agent用の環境を初期化
    env = TSPEnv(
        batch_size=inference["batch_size"], 
        n_cities=environment["num_cities"], 
        fixed_coords=input_data
    )
    env.generate_coords()
    # baseline用の環境を初期化
    baseline_env = TSPEnv(
        batch_size=inference["batch_size"], 
        n_cities=environment["num_cities"], 
        fixed_coords=input_data
    )
    baseline_env.generate_coords()

    reward_history_greedy = []
    reward_history_random = []

    # 保存済みモデルをロード
    agent.load_model(model_path)
    baseline.model.load_state_dict(copy.deepcopy(agent.model.state_dict()))

    for episode in range(episodes):
        # print('episode:', episode)
        # 1. AgentとBaselineの環境をリセット
        data, visited_cities = env.reset()
        baseline_data, visited_cities_baseline = baseline_env.reset()        

        # 2. encoder_forwardを実行
        agent.encoder_forward(data)
        baseline.encoder_forward(baseline_data)

        # doneフラグの初期化
        agent_done = False
        baseline_done = False

        random_reward = 0
        random_visit_orders = []
        greedy_reward = 0
        greedy_visit_orders = []
        
        # agentの推論
        while not agent_done:
            action, probs = agent.get_action(visited_cities)
            random_visit_orders.append(action)
            next_visited_cities, reward, agent_done = env.step(action)
            
            random_reward += reward
            visited_cities = next_visited_cities
        
        # baselineの推論
        while not baseline_done:
            action, probs = baseline.get_action(visited_cities_baseline)
            greedy_visit_orders.append(action)
            next_visited_cities, reward, baseline_done = baseline_env.step(action)
            
            greedy_reward += reward
            visited_cities_baseline = next_visited_cities
        
        # ランダムサンプリングの結果を保存
        reward_history_random.append({
            "total_reward": random_reward,
            "visit_orders": random_visit_orders
        })

        # greedyサンプリングの結果を保存
        reward_history_greedy.append({
            "total_reward": greedy_reward,
            "visit_orders": greedy_visit_orders
        })

    # データの準備
    random_reward_results = [-1 * r["total_reward"] for r in reward_history_random]
    greedy_reward_results = [-1 * r["total_reward"] for r in reward_history_greedy]

    # Best Modelでグラフをプロット
    if plot:
        utils.plot_route(input_data, reward_history_random, "random")  # ランダムサンプリングの最短経路のプロット
        utils.plot_route(input_data, reward_history_greedy, "greedy")  # Greedyサンプリングの最短経路のプロット
        utils.plot_reward_history(random_reward_results, greedy_reward_results)

    random_min_reward = min(random_reward_results)  # ランダムサンプリングの最小値
    greedy_min_reward = min(greedy_reward_results)  # Greedyサンプリングの最小値
    print(f"Random Sampling Min Reward: {random_min_reward}")
    print(f"Greedy Sampling Min Reward: {greedy_min_reward}")
    
    # optuna対応
    # print(float(min_reward.item()))  # 文字列入れない、optunaでエラーになる


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs='?', default="save/model.pth")
    parser.add_argument("--plot", action="store_true", help="プロットを表示するかどうか")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    main(args.model_path, args.episodes, plot=args.plot)