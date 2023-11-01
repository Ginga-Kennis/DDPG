import gymnasium as gym
from src.agent import Agent

# Agent parameter
STATE_SIZE = 3
ACTION_SIZE = 1
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 1e-2
SIGMA = 0.05

# Action recaling
ACTION_SCALE = 2

# learning parameters
NUM_EPISODES = 10000
PRINT_EVERY = 500

if __name__ == "__main__":
    env = gym.make("Pendulum-v1", g=9.81)
    agent = Agent(STATE_SIZE, ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, SIGMA)

    scores = []
    for i in range(1,NUM_EPISODES+1):
        state = env.reset()[0]
        score = 0
        while True:
            action = agent.act(state, add_noise = True) * ACTION_SCALE
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score += reward

            if done or truncated:
                break

        scores.append(score)
        agent.save_model("pendulum")

        if i % PRINT_EVERY == 0:
            print(f"Episode {i} : {sum(scores)/PRINT_EVERY}")
            scores.clear()

    env.close()


