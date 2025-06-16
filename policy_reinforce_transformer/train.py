import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from env.env import TSPEnv
from agent.agent import Agent
import json
import matplotlib.pyplot as plt
import numpy as np
import logging

# ログ設定
logging.basicConfig(
    filename='training.log',  # 出力先ファイル
    filemode='a',             # 'w' だと毎回上書き、'a' なら追記
    level=logging.INFO,       # ログレベル
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(lr=1e-4, gamma=0.99, episodes=10, save_path="save/model.pth"):
    env = TSPEnv(batch_size=512, n_cities=25)
    agent = Agent(lr=lr, gamma=gamma)
    reward_history = []

    for episode in range(episodes):
        # print('episode:', episode)
        data, visited_cities = env.reset()
        batch_size = visited_cities.shape[0]
        agent.encoder_forward(data)
        done = torch.zeros(batch_size, dtype=torch.bool)
        total_reward = 0
        visit_orders = []

        while not done.all():
            # print('train get_action')
            action, probs = agent.get_action(visited_cities)
            # print('probs:', probs)
            # print('action:', action)
            visit_orders.append(action)

            next_visited_cities, reward, done = env.step(action)
            # print(f"done: {done}")
            # print(f'reward: {reward}')
            # print('next_state:')
            # print(next_state)

            agent.add(reward, probs)
            total_reward += reward
            # print(f'total_reward: {total_reward}')
            visited_cities = next_visited_cities

        agent.update()
        reward_history.append(total_reward)

        # print(f'Episode: {episode}, Total Reward: {total_reward}')

        if episode % 1000 == 0:
            rewards = np.array(total_reward)
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
    plt.figure()
    plt.plot(mean_rewards, label='Mean Reward')
    plt.title("Average Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    plt.grid(True)
    plt.legend()
    # plt.show()
    png_path = save_path.replace(".pth", ".png")
    plt.savefig(png_path)
    plt.close()  # 状態をリセット（重要）

    
    # save config
    config = {
        "lr": lr,
        "gamma": gamma,
    }
    json_path = save_path.replace(".pth", ".json")
    with open(json_path, "w") as f:
        json.dump(config, f)
    # print(f"[TRAIN] Config saved at: {json_path}")

if __name__ == '__main__':
    main()
