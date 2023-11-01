import torch
import gymnasium as gym
from src.agent import Agent

# Agent parameter
STATE_SIZE = 39
ACTION_SIZE = 28

# Action recaling
ACTION_SCALE = 1

if __name__ == "__main__":
    # Agent with pretrained model
    agent = Agent(STATE_SIZE, ACTION_SIZE)
    agent.actor.load_state_dict(torch.load("./data/door_actor.pth"))
    agent.critic.load_state_dict(torch.load("./data/door_critic.pth"))

    env = gym.make('AdroitHandDoor-v1', render_mode="human", max_episode_steps=400)
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