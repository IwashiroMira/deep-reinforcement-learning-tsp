import torch
import json
import numpy as np
import matplotlib.pyplot as plt

@staticmethod
def get_device():    
    # デバイスの定義 (GPUが利用可能であればGPUを使う)
    if torch.backends.mps.is_available():  # M1/M2チップのMetal API（GPU）対応
        device = torch.device("mps")
    elif torch.cuda.is_available():  # CUDA対応（通常のGPU）
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")  # CPU
    return device

@staticmethod
def save_config(lr, gamma, save_path):
    """
    設定をJSON形式で保存する関数
    :param lr: 学習率
    :param gamma: 割引率
    :param save_path: モデルの保存パス（.pthファイル）
    """
    config = {"lr": lr, "gamma": gamma,}
    json_path = save_path.replace(".pth", ".json")
    with open(json_path, "w") as f:
        json.dump(config, f)

@staticmethod
def load_config(model_path):
    """
    モデルの設定をJSONファイルから読み込む関数
    :param model_path: モデルのパス（.pthファイル）
    :return: 設定辞書
    """
    config_path = model_path.replace(".pth", ".json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

@staticmethod
def plot_route_by_order(input_coords, orders, reward, type):
    best_reward = reward
    best_order = orders

    # 訪問順序の最初と最後に 0 を追加して巡回経路にする
    best_order.append(best_order[0])
    
    # 訪問順序に基づく座標を取得
    ordered_coords = [input_coords[i] for i in best_order]
    
    # プロットの作成
    plt.figure(figsize=(6, 6))
    plt.scatter(input_coords[:, 0], input_coords[:, 1], color='red', label='Cities')
    plt.plot(*zip(*ordered_coords), marker='o', linestyle='-', color='blue', label='Best Route')

    # 各都市にラベルを付与
    for i, (x, y) in enumerate(input_coords):
        plt.text(x, y, str(i), fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Best Route with Total Reward {type}: {best_reward}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./route/Best Route {type}")
    plt.close()  # 状態をリセット（重要）

@staticmethod
def plot_route(input_data, reward_history, type):
    # 最小トータルリワードのエピソードを取得
    best_episode = max(reward_history, key=lambda x: x["total_reward"])
    best_reward = best_episode["total_reward"]
    best_order = best_episode["visit_orders"]
    visit_order_list = [v.item() for v in best_order]
    print(f"visit_order_list: {visit_order_list}")
    print(f'best_reward: {best_reward}')

    # 訪問順序の最初と最後に 0 を追加して巡回経路にする
    best_order.append(best_order[0])
    
    # バッチ次元を除いて扱う
    input_coords = input_data[0]  # shape: (25, 2)
    
    # 訪問順序に基づく座標を取得
    ordered_coords = [input_coords[i] for i in best_order]

    # プロットの作成
    plt.figure(figsize=(6, 6))
    plt.scatter(input_coords[:, 0], input_coords[:, 1], color='red', label='Cities')
    plt.plot(*zip(*ordered_coords), marker='o', linestyle='-', color='blue', label='Best Route')

    # 各都市にラベルを付与
    for i, (x, y) in enumerate(input_coords):
        plt.text(x, y, str(i), fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Best Route with Total Reward {type}: {-1 * best_reward}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./route/Best Route {type}")
    plt.close()  # 状態をリセット（重要）

@staticmethod
def plot_reward_history(random_reward_history, greedy_reward_history):
    plt.figure(figsize=(10, 6))
    plt.plot(random_reward_history, label='random', linestyle='-', marker='o')
    plt.plot(greedy_reward_history, label='greedy', linestyle='--', marker='x')
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward History')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./route/Best Route Reward History")
    plt.close()  # 状態をリセット（重要）

@staticmethod
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





