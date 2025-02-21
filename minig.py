import random
import minigrid
import gymnasium

env = gymnasium.make("MiniGrid-Empty-5x5-v0", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
    a = env.render()
    print(a)
    action = random.randint(0, env.action_space.n-1)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()