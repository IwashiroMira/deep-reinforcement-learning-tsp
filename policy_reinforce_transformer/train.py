import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import TSPEnv
from agent.agent import Agent
import json
import matplotlib.pyplot as plt

def main(lr=1e-4, gamma=0.99, episodes=100, save_path="model.pth"):
    env = TSPEnv()
    agent = Agent(lr=lr, gamma=gamma)
    reward_history = []

    for episode in range(episodes):
        # print('episode:', episode)
        data, visited_cities = env.reset()
        agent.encoder_forward(data)
        done = False
        total_reward = 0
        visit_orders = []

        while not done:
            # print('train get_action')
            action, probs = agent.get_action(visited_cities)
            # print('probs:', probs)
            # print('action:', action)
            visit_orders.append(action)

            next_visited_cities, reward, done = env.step(action)
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

        if episode % 500 == 0:
            print(f'Episode: {episode}, Total Reward: {total_reward}')
            print(f'visit_orders:{visit_orders}')
    
    # save model
    agent.save_model(save_path)
    # print(f"[TRAIN] Model saved at: {save_path}")

    # --- ここからプロット部分 ---
    plt.plot(reward_history)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

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