import time
import gymnasium as gym
from stable_baselines3 import PPO

# load the saved model
model = PPO.load("models/ppo_cartpole")

# recreate the environment
env = gym.make("CartPole-v1", render_mode="human")

# TODO: record video but it is only possible in a different render mode
# env = gym.wrappers.RecordVideo(
#     env,
#     video_folder="videos/",
#     episode_trigger=lambda episode_id: True,  # record every episode
#     name_prefix="ppo_cartpole"
# )

obs = env.reset()[0]  # get the initial observation

for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    time.sleep(0.02)  # slow down so it's visible

    if done or truncated:
        obs = env.reset()[0]  # restart the episode

env.close()




