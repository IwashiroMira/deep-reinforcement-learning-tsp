import sys
import os
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import TSPEnv
from agent.agent import Agent

def main():
    episodes = 500
    env = TSPEnv()
    agent = Agent()
    reward_history = []

    for episode in range(episodes):
        # print('episode:', episode)
        state = env.reset()
        done = False
        total_reward = 0
        visit_orders = []

        while not done:
            action, probs = agent.get_action(state)
            # print('probs:', probs)
            # print('action:', action)
            visit_orders.append(action)

            next_state, reward, done = env.step(action)
            # print(f'reward: {reward}')
            # print('next_state:')
            # print(next_state)

            agent.add(reward, probs)
            total_reward += reward
            state = next_state

        agent.update()
        reward_history.append(total_reward)

        # print(f'Episode: {episode}, Total Reward: {total_reward}')

        if episode % 50 == 0:
            print(f'Episode: {episode}, Total Reward: {total_reward}')
            print(state)
            print(f'visit_orders:{visit_orders}')
    
    # save model
    path = 'save/model.pth'
    agent.save_model(path)

if __name__ == '__main__':
    main()