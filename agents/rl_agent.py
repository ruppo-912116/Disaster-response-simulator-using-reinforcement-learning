from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os

class RLAgent:
    def __init__(self, env, model_path="models/ppo_disaster"):
        self.env = env
        self.model_path = model_path
        self.model = None
        
    def create_model(self):
        """Initialize the PPO model."""
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        
    def train(self, total_timesteps=100000, eval_freq=10000):
        """Train the model with evaluation callbacks."""
        # Create evaluation environment
        eval_env = self.env
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_path,
            log_path=self.model_path,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save the final model
        self.save_model()
        
    def save_model(self):
        """Save the trained model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
    def load_model(self):
        """Load a trained model."""
        self.model = PPO.load(self.model_path, env=self.env)
        
    def predict(self, observation):
        """Predict the best action for a given observation."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.predict(observation, deterministic=True)
        
    def evaluate(self, n_episodes=10):
        """Evaluate the model's performance."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        episode_rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.predict(obs)
                obs, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                
            episode_rewards.append(episode_reward)
            
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards)
        } 