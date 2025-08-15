# Disaster Response Simulator with Reinforcement Learning

A simulation environment where a Reinforcement Learning agent learns to respond to natural disasters in a virtual city. The agent observes the state of the city and decides on actions such as deploying firetrucks, evacuating people, or opening shelters to minimize casualties and damage.

## 🚀 Features

- Custom Gymnasium environment for disaster simulation
- PPO-based RL agent for learning optimal response strategies
- Real-time visualization of disaster spread and agent actions
- Training metrics and evaluation tools
- Configurable grid size and disaster parameters

## 📋 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/disaster-rl.git
cd disaster-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. Train the agent:
```bash
python main.py
```

This will:
- Create and train a PPO agent
- Save the trained model
- Evaluate the agent's performance
- Visualize sample episodes
- Plot training metrics

## 📁 Project Structure

```
disaster_rl/
├── main.py                      # Entry point
├── environment/
│   └── disaster_env.py          # Gym environment
├── agents/
│   └── rl_agent.py              # PPO training wrapper
├── utils/
│   └── visualization.py         # Grid visualizer
├── models/                      # Saved RL models
└── README.md
```

## 🔧 Environment Details

The environment consists of a grid where each cell contains:
- Disaster intensity (0-1)
- Population density (0-1)
- Infrastructure status (0-1)

### Actions
- 0: No operation
- 1: Deploy firetruck
- 2: Evacuate population
- 3: Open shelter

### Rewards
The agent receives rewards based on:
- Negative reward for high disaster intensity
- Negative reward for population in disaster areas
- Positive reward for maintaining infrastructure

## 📊 Results

The agent learns to:
- Quickly respond to disaster outbreaks
- Prioritize high-risk areas
- Balance resource allocation
- Minimize overall damage and casualties

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 