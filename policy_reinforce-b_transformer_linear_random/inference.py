import sys
import os
import numpy as np
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
import torch


def get_minimum_reward(reward_history):
    """
    各エピソードにおける total_reward のうち、最小の報酬を持つエントリを探し、
    その報酬、訪問順序、座標を返す。
    """
    min_episode_reward = float("inf")
    best_order = None
    coords = None
    best_episode = None
    best_idx = None

    for episode, data in enumerate(reward_history):
        # 最小報酬のインデックスを取得
        min_idx = data["total_reward"].argmin()
        current_reward = data["total_reward"][min_idx]

        if current_reward < min_episode_reward:
            min_episode_reward = current_reward

            # visit_orders: (visit_step, batch_size) → 転置して (batch_size, visit_step)
            visit_tensor_T = torch.stack(data["visit_orders"]).T
            best_order = visit_tensor_T[min_idx]

            # 対応する座標 (25, 2)
            coords = data["coords"][min_idx]

            # インデックス記録
            best_episode = episode
            best_idx = min_idx

    return min_episode_reward, best_order, coords, best_episode, best_idx


def main(model_path='save/model.pth', episodes=100, plot=True, fixed=True):
    # inputファイルの読み込み
    if fixed:
        loaded_data = np.load(f"data/fixed_coords_{environment['num_cities']}.npy")
        # B×N×D に reshape（B=1）
        input_data = loaded_data[np.newaxis, :, :]  # shape = (1, 25, 2)

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

    if fixed:
        # agent用の環境を初期化
        env = TSPEnv(
            batch_size=inference["batch_size"], 
            n_cities=environment["num_cities"], 
            fixed_coords=input_data
        )
    else:
        # agent用の環境を初期化
        env = TSPEnv(
            batch_size=inference["batch_size"], 
            n_cities=environment["num_cities"], 
        )

    # 保存済みモデルをロード
    agent.load_model(model_path)
    baseline.model.load_state_dict(copy.deepcopy(agent.model.state_dict()))

    # 記録用
    reward_history_greedy = []
    reward_history_random = []

    for episode in range(episodes):
        # print('episode:', episode)
        # 1. 環境の座標を生成
        env.generate_coords()  # 都市の座標を生成
        # 2. Baseline用の環境（同じ座標を固定）
        baseline_env = TSPEnv(
            batch_size=inference["batch_size"], 
            n_cities=environment["num_cities"], 
            fixed_coords=env.coords.copy()  # 現在の座標をコピー
        )
        baseline_env.generate_coords()  # 固定された座標を使用して環境を初期化        

        # 1. AgentとBaselineの環境をリセット
        data, visited_cities = env.reset()
        baseline_data, visited_cities_baseline = baseline_env.reset()        

        # 2. encoder_forwardを実行
        agent.encoder_forward(data)
        baseline.encoder_forward(baseline_data)

        # doneフラグの初期化
        batch_size = visited_cities.shape[0]
        agent_done = torch.zeros(batch_size, dtype=torch.bool)
        baseline_done = torch.zeros(batch_size, dtype=torch.bool)

        random_reward = 0
        random_visit_orders = []
        greedy_reward = 0
        greedy_visit_orders = []
        
        # agentの推論
        while not agent_done.all():
            action, probs = agent.get_action(visited_cities)
            random_visit_orders.append(action)
            next_visited_cities, reward, agent_done = env.step(action)
            
            random_reward += reward
            visited_cities = next_visited_cities
        
        # baselineの推論
        while not baseline_done.all():
            action, probs = baseline.get_action(visited_cities_baseline)
            greedy_visit_orders.append(action)
            next_visited_cities, reward, baseline_done = baseline_env.step(action)
            
            greedy_reward += reward
            visited_cities_baseline = next_visited_cities
        
        # ランダムサンプリングの結果を保存
        reward_history_random.append({
            "total_reward": -1 * random_reward,
            "visit_orders": random_visit_orders,
            "coords": env.coords
        })

        # greedyサンプリングの結果を保存
        reward_history_greedy.append({
            "total_reward": -1 * greedy_reward,
            "visit_orders": greedy_visit_orders
        })

    # print(f"reward_history_random: {reward_history_random}")
    # print(f"reward_history_greedy: {reward_history_greedy}")
    
    # randomのベストエピソードを取得
    random_min_reward, random_best_order, coords, best_episode, best_idx = get_minimum_reward(reward_history_random)
    random_best_order = random_best_order.cpu().tolist()  # MPS -> CPU & list化

    # randomのエピソードとインデックス番号からgreedyのベストエピソードを取得
    greedy_best_episode = reward_history_greedy[best_episode]
    greedy_visited_tensor_T = torch.stack(greedy_best_episode["visit_orders"]).T
    greedy_best_order = greedy_visited_tensor_T[best_idx].cpu().tolist()  # Greedyの訪問順序を取得

    # greedy_best_order = greedy_best_episode["visit_orders"][best_idx].cpu().tolist()  # MPS -> CPU & list化
    greedy_min_reward = greedy_best_episode["total_reward"][best_idx].item()  # MPS -> CPU & float化
    print(f"Best Episode: {best_episode}, Best Index: {best_idx}")
    print(f"Random Sampling Min Reward: {random_min_reward}")
    print(f"Greedy Sampling Min Reward: {greedy_min_reward}")
    print(f"Random Best Order: {random_best_order}")
    print(f"Greedy Best Order: {greedy_best_order}")
    print(f"Coordinates: {coords}")

    # 各エピソードにおける報酬の平均化準備
    random_reward_results = [r["total_reward"].mean() for r in reward_history_random]
    greedy_reward_results = [r["total_reward"].mean() for r in reward_history_greedy]

    # Best Modelでグラフをプロット
    if plot:
        utils.plot_route_by_order(coords, random_best_order, random_min_reward, "Random")  # ランダムサンプリングの最短経路のプロット
        utils.plot_route_by_order(coords, greedy_best_order, greedy_min_reward, "Greedy")  # Greedyサンプリングの最短経路のプロット
        utils.plot_reward_history(random_reward_results, greedy_reward_results)

    # optuna対応
    # print(float(min_reward.item()))  # 文字列入れない、optunaでエラーになる


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("model_path", nargs='?', default="save/model.pth")
    # parser.add_argument("--plot", action="store_true", help="プロットを表示するかどうか")
    # parser.add_argument("--episodes", type=int, default=3)
    # args = parser.parse_args()
    # main(args.model_path, args.episodes, plot=args.plot)

    model_path = inference["model_path"]
    episodes = inference["episodes"]
    plot = inference["plot"]
    fixed = inference["fixed"]
    main(model_path=model_path, episodes=episodes, plot=plot, fixed=fixed)