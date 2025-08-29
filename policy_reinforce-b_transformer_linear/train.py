import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from env.env import TSPEnv
from agent.agent import Agent
from agent.baseline import BaseLine
import matplotlib.pyplot as plt
from utils import save_config
import numpy as np
import logging
import copy
from scipy.stats import ttest_rel
from tqdm import tqdm
from config import environment, training


# ログ設定
logging.basicConfig(
    filename='training.log',  # 出力先ファイル
    filemode='a',             # 'w' だと毎回上書き、'a' なら追記
    level=logging.INFO,       # ログレベル
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(lr=training["lr"], 
         gamma=training["gamma"], 
         episodes=training["episodes"], 
         save_path=training["model_path"]
         ):
    env = TSPEnv(batch_size=training["batch_size"], n_cities=environment["num_cities"])
    agent = Agent(lr=lr, gamma=gamma)
    baseline = BaseLine(lr=lr, gamma=gamma)
    baseline.model.load_state_dict(copy.deepcopy(agent.model.state_dict()))
    reward_history = []
    baseline_reward_history = []

    for episode in range(episodes):
        # print('episode:', episode)
        # 1. 環境の座標を生成
        env.generate_coords()  # 都市の座標を生成
        # 2. Baseline用の環境（同じ座標を固定）
        fixed_coords = env.coords.copy()  # 現在の座標をコピー
        baseline_env = TSPEnv(batch_size=training["batch_size"], n_cities=environment["num_cities"], fixed_coords=fixed_coords)
        baseline_env.generate_coords()  # 固定された座標を使用して環境を初期化

        # 3. AgentとBaselineの環境をリセット
        data, visited_cities = env.reset()
        baseline_data, visited_cities_baseline = baseline_env.reset()        

        batch_size = visited_cities.shape[0]
        # encoder_forwardを実行
        agent.encoder_forward(data)
        baseline.encoder_forward(baseline_data)

        agent_done = torch.zeros(batch_size, dtype=torch.bool)
        baseline_done = torch.zeros(batch_size, dtype=torch.bool)
        agent_total_reward = 0
        baseline_total_reward = 0
        # agent_visit_orders = []

        while not agent_done.all():
            # print('train get_action')
            action, probs = agent.get_action(visited_cities)
            # agent_visit_orders.append(action)
            # 状態更新
            next_visited_cities, reward, agent_done = env.step(action)
            # 報酬と確率をエージェントに追加
            agent.add(reward, probs)
            agent_total_reward += reward
            # print(f'total_reward: {total_reward}')
            visited_cities = next_visited_cities

        while not baseline_done.all():
            # print('train get_action')
            action, probs = baseline.get_action(visited_cities_baseline)
            # visit_orders.append(action)
            # 状態更新
            baseline_next_visited_cities, baseline_reward, baseline_done = baseline_env.step(action)
            # 報酬を更新
            baseline_total_reward += baseline_reward
            visited_cities_baseline = baseline_next_visited_cities

        # パラメータ更新
        agent.update(baseline_total_reward)

        # t検定
        t, p = ttest_rel(agent_total_reward, baseline_total_reward)
        if t > 0 and (p / 2) < 0.05:
            logging.info("Agent's performance is significantly better than the baseline.")
            logging.info(f'agent_total_reward: {agent_total_reward.mean()}, baseline_total_reward: {baseline_total_reward.mean()}')
            logging.info(f't-statistic: {t}, p-value: {p}')
            baseline.model.load_state_dict(copy.deepcopy(agent.model.state_dict()))
        # t_stat, p_value = ttest_ind(agent_total_reward, baseline_total_reward)

        reward_history.append(agent_total_reward)
        baseline_reward_history.append(baseline_total_reward)

        # print(f'Episode: {episode}, Total Reward: {total_reward}')
        if episode % 1000 == 0:
            rewards = np.array(agent_total_reward)
            mean = rewards.mean()
            std = rewards.std()
            min_ = rewards.min()
            max_ = rewards.max()            
            print(f"Episode: {episode}, Mean: {mean:.3f}, Std: {std:.3f}, Min: {min_}, Max: {max_}")
            log_msg = f"Episode: {episode}, Mean: {mean:.3f}, Std: {std:.3f}, Min: {min_}, Max: {max_}"
            logging.info(log_msg)     # ログファイルに記録
    
    # save model
    agent.save_model(save_path)
    # print(f"[TRAIN] Model saved at: {save_path}")

    # --- ここからプロット部分 ---
    reward_history = np.array(reward_history)  # shape: (episodes, batch_size)
    mean_rewards = reward_history.mean(axis=1)
    baseline_reward_history = np.array(baseline_reward_history)
    baseline_mean_rewards = baseline_reward_history.mean(axis=1)
    plt.figure()
    plt.plot(mean_rewards, label='Mean Reward')
    plt.plot(baseline_mean_rewards, label='BaseLine Mean Reward')
    plt.title("Average Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    plt.grid(True)
    plt.legend()
    # plt.show()
    png_path = save_path.replace(".pth", ".png")
    plt.savefig(png_path)
    plt.close()  # 状態をリセット（重要）
    
    # save config by utils.save_config
    save_config(lr, gamma, save_path)


if __name__ == '__main__':
    main()
