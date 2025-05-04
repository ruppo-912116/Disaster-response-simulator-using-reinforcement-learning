import matplotlib.pyplot as plt
import numpy as np

def visualize_state(state, title="Disaster Environment State"):
    """Visualize the current state of the environment."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Disaster intensity
    im0 = axes[0].imshow(state[:, :, 0], cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title('Disaster Intensity')
    plt.colorbar(im0, ax=axes[0])
    
    # Population density
    im1 = axes[1].imshow(state[:, :, 1], cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Population Density')
    plt.colorbar(im1, ax=axes[1])
    
    # Infrastructure
    im2 = axes[2].imshow(state[:, :, 2], cmap='Greens', vmin=0, vmax=1)
    axes[2].set_title('Infrastructure')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def visualize_episode(env, agent, max_steps=100):
    """Visualize a complete episode of the agent interacting with the environment."""
    obs, _ = env.reset()
    done = False
    step = 0
    
    # Create figure for animation
    fig = plt.figure(figsize=(15, 5))
    
    while not done and step < max_steps:
        # Get agent's action
        action, _ = agent.predict(obs)
        
        # Step the environment
        obs, reward, done, _, _ = env.step(action)
        
        # Visualize current state
        plt.clf()
        visualize_state(obs, title=f'Step {step}, Action: {action}, Reward: {reward:.2f}')
        plt.pause(0.1)
        
        step += 1
    
    plt.close()
    
def plot_training_metrics(log_path):
    """Plot training metrics from the log file."""
    import pandas as pd
    
    # Read the log file
    df = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot mean reward
    axes[0].plot(df['timesteps'], df['mean_reward'])
    axes[0].set_title('Mean Reward')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Reward')
    
    # Plot episode length
    axes[1].plot(df['timesteps'], df['ep_len_mean'])
    axes[1].set_title('Mean Episode Length')
    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Steps')
    
    plt.tight_layout()
    return fig 