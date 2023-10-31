import gymnasium as gym
from src.agent import Agent

ACTION_SCALE = 2

if __name__ == "__main__":
    env = gym.make("Pendulum-v1", g=9.81)
    agent = Agent(state_size=3, action_size=1)

    n_episodes = 1000
    max_t = 300
    print_every = 50

    scores = []
    for i in range(1,n_episodes+1):
        state = env.reset()[0]
        agent.reset()  # reset OUnoise

        score = 0
        for _ in range(max_t):
            action = agent.act(state) * ACTION_SCALE
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        agent.save_model("pendulum")

        if i % print_every == 0:
            print(f"Episode {i} : {sum(scores)/print_every}")
            scores.clear()

    env.close()


