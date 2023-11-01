import torch
import gymnasium as gym
from src.agent import Agent

ACTION_SCALE = 2

if __name__ == "__main__":
    # Agent with pretrained model
    agent = Agent(state_size=3, action_size=1)
    agent.actor.load_state_dict(torch.load("./data/pendulum_actor.pth"))
    agent.critic.load_state_dict(torch.load("./data/pendulum_critic.pth"))

    env = gym.make("Pendulum-v1", g=9.81, render_mode="human")
    state = env.reset()[0]

    score = 0
    while True:
        action = agent.act(state, add_noise=False) * ACTION_SCALE
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        score += reward
        if done or truncated:
            break

    print(score)
    env.close()



