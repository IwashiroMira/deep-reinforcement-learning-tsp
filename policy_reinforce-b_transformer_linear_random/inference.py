import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from env.env import TSPEnv
from agent.agent import Agent
from agent.baseline import BaseLine
from model.exact_model import ExactModel
import copy
import argparse
import sys
from config import environment, inference
import utils
import torch
import time
import logging

# ログ設定
logging.basicConfig(
    filename='inference.log',  # 出力先ファイル
    filemode='a',             # 'w' だと毎回上書き、'a' なら追記
    level=logging.INFO,       # ログレベル
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_best_orders(random_history, greedy_history):
    """
    ランダムサンプリングとGreedyサンプリングの結果から、最小報酬を持つ訪問順序を抽出する。
    :param random_history: ランダムサンプリングの結果
    :param greedy_history: Greedyサンプリングの結果
    :return: 最小報酬と訪問順序
    """
    random_min_reward, random_best_order, coords, best_episode, best_idx = get_minimum_reward(random_history)
    random_best_order = random_best_order.cpu().tolist()  # MPS -> CPU & list化

    # randomのエピソードとインデックス番号からgreedyのベストエピソードを取得
    greedy_episode = greedy_history[best_episode]
    greedy_tensor = torch.stack(greedy_episode["visit_orders"]).T
    greedy_order = greedy_tensor.cpu().tolist()[0]  # Greedyの訪問順序を取得
    greedy_reward = greedy_episode["total_reward"].item()  # MPS -> CPU & float化

    return {
        "random_min_reward": random_min_reward,
        "random_best_order": random_best_order,
        "coords": coords,
        "best_episode": best_episode,
        "best_idx": best_idx,
        "greedy_reward": greedy_reward,
        "greedy_order": greedy_order
    }

def execute_exact_model(coords):
    """
    ExactModelを使用して、与えられた座標に対する最適な巡回路を計算する。
    :param coords: 都市の座標 (batch_size, num_cities, coord_dim)
    :return: 最適な巡回路の訪問順序とトータル距離
    """
    # ExactModelのインスタンスを作成
    exact_model = ExactModel(num_cities=environment["num_cities"], coords=coords)

    # 距離行列を計算
    exact_model.calculate_distance_matrix()

    # モデルを実行して最適な巡回路を求める
    exact_model.execute()

    # 最適な巡回路の訪問順序とトータル距離を返す
    return exact_model.total_distance, exact_model.visit_order

def load_input_data(fixed, num_cities):
    """
    入力データを読み込み関数。
    :param fixed: 座標が固定されているかどうか
    :param num_cities: 都市の数
    :return: 座標データ
    """
    # 座標が固定されていない場合は None を返す
    if not fixed:
        return None
    # 固定座標のデータを読み込む
    data = np.load(f"data/fixed_coords_{num_cities}.npy")
    # B×N×D に reshape（B=1）
    return data[np.newaxis, :, :]  # shape = (1, 25, 2)

def create_env(fixed_coords, batch_size, num_cities):
    """
    環境を生成するヘルパー関数。
    :param fixed: 座標が固定されているかどうか
    :param num_cities: 都市の数
    :return: TSPEnvインスタンス
    """
    return TSPEnv(
        batch_size=batch_size, 
        n_cities=num_cities, 
        fixed_coords=fixed_coords
    )

def run_inference_episode(baseline_env, agent, baseline, n_cities, batch_size):
    """
    1エピソードの推論を実行する。
    :param env: TSPEnvインスタンス
    :param agent: Agentインスタンス
    :param baseline: BaseLineインスタンス
    :param n_cities: 都市の数
    :param batch_size: バッチサイズ
    :return: 推論結果（報酬、訪問順序）
    """
    # print("run_inference_episode")
    # 1. 環境の座標を生成
    baseline_env.generate_coords("baseline")  # 都市の座標を生成
    # 2. Baseline用の環境（同じ座標を固定）
    env = create_env(
        fixed_coords=baseline_env.coords.copy(),
        batch_size=batch_size,
        num_cities=n_cities
    )
    env.generate_coords("random")  # 固定された座標を使用して環境を初期化

    # logging.info(f"env.coords: {env.coords.shape}, baseline_env.coords: {baseline_env.coords.shape}")

    # 1. AgentとBaselineの環境をリセット
    data, visited_cities = env.reset()
    baseline_data, visited_cities_baseline = baseline_env.reset()        

    # 2. encoder_forwardを実行
    agent.encoder_forward(data)
    baseline.encoder_forward(baseline_data)

    # doneフラグの初期化
    batch = visited_cities.shape[0]
    agent_done = torch.zeros(batch, dtype=torch.bool)
    baseline_done = torch.zeros(batch, dtype=torch.bool)

    random_reward, greedy_reward = 0, 0
    random_visit_orders, greedy_visit_orders = [], []
    
    # agentの推論
    while not agent_done.all():
        action, _ = agent.get_action(visited_cities)
        random_visit_orders.append(action)
        visited_cities, reward, agent_done = env.step(action)
        random_reward += reward
        
    # baselineの推論
    while not baseline_done.all():
        action, probs = baseline.get_action(visited_cities_baseline)
        greedy_visit_orders.append(action)
        visited_cities_baseline, reward, baseline_done = baseline_env.step(action)
        greedy_reward += reward
    
    return {
        "random": {
            "total_reward": -1 * random_reward,
            "visit_orders": random_visit_orders,
            "coords": env.coords
        },
        "greedy": {
            "total_reward": -1 * greedy_reward,
            "visit_orders": greedy_visit_orders,
            "coords": baseline_env.coords
        }
    }

def get_minimum_reward(random_result):
    """
    各エピソードにおける total_reward のうち、最小の報酬を持つエントリを探し、
    その報酬、訪問順序、座標を返す。
    """
    # 最小報酬のインデックスを取得
    min_idx = random_result["total_reward"].argmin()
    min_reward = random_result["total_reward"][min_idx]

    # visit_orders: (visit_step, batch_size) → 転置して (batch_size, visit_step)
    visit_tensor_T = torch.stack(random_result["visit_orders"]).T
    best_order = visit_tensor_T[min_idx]

    return min_reward, best_order

def calc_reward_mean(history):
    rewards = np.array([float(h["total_reward"]) for h in history], dtype=float)
    return float(rewards.mean())


def print_results(info, exact_reward, exact_best_order,):
    """
    結果をコンソールに出力する。
    :param best_info: 最良の結果情報
    """
    logging.info(f"Best Episode: {info["best_episode"]}, Best Index: {info["best_idx"]}")
    logging.info(f"Exact Reward: {exact_reward}")
    logging.info(f"Random Reward: {info["random_min_reward"]}")
    logging.info(f"Greedy Reward: {info["greedy_reward"]}")
    logging.info(f"Exact Order: {exact_best_order}")
    logging.info(f"Random Order: {info["random_best_order"]}")
    logging.info(f"Greedy Order: {info["greedy_order"]}")
    # print(f"Coordinates: {info["coords"]}")

def plot_results(info, exact_reward, exact_best_order, random_history, greedy_history):
    coords = info["coords"]
    utils.plot_route_by_order(coords, info["random_best_order"], info["random_min_reward"], "Random")
    utils.plot_route_by_order(coords, info["greedy_order"], info["greedy_reward"], "Greedy")
    utils.plot_route_by_order(coords, exact_best_order, exact_reward, "Exact")

    random_avg = [r["total_reward"].mean() for r in random_history]
    greedy_avg = [r["total_reward"].mean() for r in greedy_history]
    utils.plot_reward_history(random_avg, greedy_avg)

def plot_reward_history(random, greedy, exact, title):
    random_avg = [r["total_reward"] for r in random]
    greedy_avg = [r["total_reward"] for r in greedy]
    exact_avg = [r["total_reward"] for r in exact]
    utils.plot_reward_history_triple(random_avg, greedy_avg, exact_avg, title)
    

def main(model_path='save/model.pth', episodes=100, plot=True, fixed=True):
    num_cities = environment["num_cities"]
    random_batch_size = inference["random_batch_size"]
    baseline_batch_size = inference["baseline_batch_size"]

    logging.info(f"推論実行：都市数 {num_cities}, 固定座標 {fixed}")

    # input_dataの読み込み, なければ None
    input_data = load_input_data(fixed, num_cities)

    # 同名の .json ファイルからパラメータ読み込み
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found.")
    config = config = utils.load_config(model_path)

    # config を使ってAgentとbaseline構築
    agent = Agent(
        lr=config["lr"],
        gamma=config["gamma"]
    )
    baseline = BaseLine(
        lr=config["lr"],
        gamma=config["gamma"]
    )

    # 環境の生成
    baseline_env = create_env(input_data, baseline_batch_size, num_cities)

    # 保存済みモデルをロード
    agent.load_model(model_path)
    baseline.model.load_state_dict(copy.deepcopy(agent.model.state_dict()))

    # 記録用
    greedy_history, random_history, exact_history = [], [], []

    # ランダムサンプリングの処理時間計測開始
    # start_loop = time.time()
    
    for episode in range(episodes):
        if episode % 100 == 0:
            logging.info(f"episode: {episode}")
        
        # 推論実行
        result = run_inference_episode(baseline_env, agent, baseline, num_cities, random_batch_size)
        
        # ランダムサンプリングの最小値を求める
        random_min_reward, random_best_order = get_minimum_reward(result["random"])
    
        # ランダムサンプリングの最小値を保存
        random_history.append({
            "total_reward": random_min_reward,
            "visit_orders": random_best_order,
        })

        # greedyサンプリングの結果を保存
        greedy_history.append({
            "total_reward": result["greedy"]["total_reward"],
            "visit_orders": result["greedy"]["visit_orders"],
            "coords": result["greedy"]["coords"]
        })

        # 厳密解を求める
        exact_reward, exact_best_order = execute_exact_model(result["greedy"]["coords"].squeeze())

        # 厳密解の結果を保存
        exact_history.append({
            "total_reward": exact_reward,
            "visit_orders": exact_best_order,
        })

    # print(f"reward_history_random: {random_history}")
    # print(f"reward_history_greedy: {greedy_history}")
    # print(f"reward_history_exact: {exact_history}")
    
    random_mean = calc_reward_mean(random_history)
    greedy_mean = calc_reward_mean(greedy_history)
    exact_mean = calc_reward_mean(exact_history)
    logging.info(f"random平均値: {random_mean}")
    logging.info(f"greedy平均値: {greedy_mean}")
    logging.info(f"exact平均値: {exact_mean}")

    # exactに対する増加率を計算
    random_rate = (random_mean - exact_mean) / exact_mean
    greedy_rate = (greedy_mean - exact_mean) / exact_mean
    
    logging.info(f"ランダムの増加率：{random_rate*100:.2f}%")
    logging.info(f"貪欲法の増加率：{greedy_rate*100:.2f}%")
    
    # # Best Modelでグラフをプロット
    if plot:
        plot_reward_history(random_history, greedy_history, exact_history, "Reward History 25")


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