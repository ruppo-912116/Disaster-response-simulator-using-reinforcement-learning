import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DisasterEnv(gym.Env):
    """Custom environment for disaster response simulation."""
    
    def __init__(self, grid_size=10):
        super(DisasterEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.max_steps = 100
        
        # Define action space
        # Actions: 0=noop, 1=deploy_firetruck, 2=evacuate, 3=open_shelter
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        # Observation: grid of cells (each cell has: disaster intensity, population, infrastructure)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 3),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.current_step = 0
        
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize empty grid
        self.state = np.zeros((self.grid_size, self.grid_size, 3))
        
        # Randomly place initial disaster
        disaster_x = np.random.randint(0, self.grid_size)
        disaster_y = np.random.randint(0, self.grid_size)
        self.state[disaster_x, disaster_y, 0] = 1.0  # Disaster intensity
        
        # Randomly place population
        self.state[:, :, 1] = np.random.random((self.grid_size, self.grid_size)) * 0.5
        
        # Initialize infrastructure (all cells start with full infrastructure)
        self.state[:, :, 2] = 1.0
        
        self.current_step = 0
        return self.state, {}
    
    def step(self, action):
        """Execute one time step within the environment."""
        self.current_step += 1
        
        # Apply action effects
        reward = self._apply_action(action)
        
        # Update disaster spread
        self._update_disaster()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        done = self.current_step >= self.max_steps
        
        return self.state, reward, done, False, {}
    
    def _apply_action(self, action):
        """Apply the selected action to the environment."""
        reward = 0
        
        if action == 1:  # Deploy firetruck
            # Find cell with highest disaster intensity
            max_intensity = np.unravel_index(
                np.argmax(self.state[:, :, 0]),
                self.state[:, :, 0].shape
            )
            # Reduce disaster intensity in that cell
            self.state[max_intensity[0], max_intensity[1], 0] *= 0.5
            reward += 0.1
            
        elif action == 2:  # Evacuate
            # Find cell with highest population and disaster intensity
            risk = self.state[:, :, 0] * self.state[:, :, 1]
            max_risk = np.unravel_index(np.argmax(risk), risk.shape)
            # Reduce population in that cell
            self.state[max_risk[0], max_risk[1], 1] *= 0.5
            reward += 0.2
            
        elif action == 3:  # Open shelter
            # Find cell with lowest disaster intensity
            min_intensity = np.unravel_index(
                np.argmin(self.state[:, :, 0]),
                self.state[:, :, 0].shape
            )
            # Increase infrastructure in that cell
            self.state[min_intensity[0], min_intensity[1], 2] = 1.0
            reward += 0.1
            
        return reward
    
    def _update_disaster(self):
        """Update disaster spread and intensity."""
        # Simple diffusion model for disaster spread
        kernel = np.array([[0.1, 0.2, 0.1],
                          [0.2, 0.4, 0.2],
                          [0.1, 0.2, 0.1]])
        
        # Apply convolution to disaster intensity
        from scipy.ndimage import convolve
        self.state[:, :, 0] = convolve(self.state[:, :, 0], kernel, mode='constant')
        
        # Normalize disaster intensity
        self.state[:, :, 0] = np.clip(self.state[:, :, 0], 0, 1)
    
    def _calculate_reward(self):
        """Calculate reward based on current state."""
        # Penalize high disaster intensity
        disaster_penalty = -np.mean(self.state[:, :, 0])
        
        # Penalize high population in disaster areas
        population_risk = np.mean(self.state[:, :, 0] * self.state[:, :, 1])
        
        # Reward high infrastructure
        infrastructure_reward = np.mean(self.state[:, :, 2])
        
        return disaster_penalty - population_risk + infrastructure_reward 