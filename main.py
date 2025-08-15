import os
from environment.disaster_env import DisasterEnv
from agents.rl_agent import RLAgent
from utils.visualization import visualize_state, visualize_episode, plot_training_metrics

def main():
    # Create environment
    env = DisasterEnv(grid_size=10)
    
    # Create and train agent
    agent = RLAgent(env)
    agent.create_model()
    
    # Train the agent
    print("Starting training...")
    agent.train(total_timesteps=100000, eval_freq=10000)
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    metrics = agent.evaluate(n_episodes=10)
    print(f"Evaluation metrics: {metrics}")
    
    # Visualize a sample episode
    print("\nVisualizing sample episode...")
    visualize_episode(env, agent)
    
    # Plot training metrics
    print("\nPlotting training metrics...")
    log_path = os.path.join("models/ppo_disaster", "evaluations.npz")
    if os.path.exists(log_path):
        plot_training_metrics(log_path)
        plt.show()

if __name__ == "__main__":
    main() 