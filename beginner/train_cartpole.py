import gymnasium as gym
from stable_baselines3 import PPO

# create the environment
env = gym.make("CartPole-v1")

# initialize PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# train the agent
model.learn(total_timesteps=10000)

# save the model
model.save("models/ppo_cartpole")

env.close()


