# Lunar lander efforts

# (A) DQN network with Episilon Greedy (best)

The working version was with DQN networks. This trains quickly and works. It uses Epsilon greedy to make it explore aggressively at the start and then less so as it gets better. It remembers what it has done.

## Prerequisites

Before running the code, you need to install FFMPEG to make videos:

```bash
sudo apt install ffmpeg
```

## Environment Setup

1. Build the conda environment:
```bash
conda env create -f dqn_environment.yml
```

2. Activate the environment:
```bash
conda activate deep_rl_v2
```

3. Install gym[box2d]:
```bash
pip install gym[box2d]
```

## Running the Training

To train the agent:
```bash
python dqn_epsilon_lander.py
```

# (B) Making A2C Work with LunarLander

This repository contains an implementation of the Advantage Actor-Critic (A2C) algorithm for training an agent to solve the LunarLander-v2 environment. There was a failed attempt using the DQN code that works with the (A) environment. If you are curious then try:

To train the agent:
```bash
python a2c-epsilon_lander.py
```

Alternatively, try this one which is based on an A2C script from Deep-Reinforcement-Learning-Hands-On-Third-Edition. It is not as good as the DQN option but homework is homework...

## Prerequisites

Before running the code, you need to install SWIG:

```bash
sudo apt install swig
```

## Environment Setup

1. Build the conda environment:
```bash
conda env create -f lunar1.yml
```

2. Activate the environment:
```bash
conda activate lunar1
```

3. Install tensorboard if you wish:
```bash
conda install tensorflow-base tensorboard
```

## Project Structure

Ensure your project directory is structured as follows:

```
your_project_directory/
├── lunar-lander-a2c.py
├── lunar1.yml
├── lib/
│   ├── __init__.py
│   ├── model.py
│   └── common.py
```

## Running the Training

To train the agent using GPU:
```bash
python lunar-lander-a2c.py -n lunar_lander --dev cuda
```

To train using CPU:
```bash
python lunar-lander-a2c.py -n lunar_lander --dev cpu
```

## Training Output

The script will:
- Create a 'saves' directory for storing the best models
- Create TensorBoard logs for monitoring training progress
- Print periodic updates about training performance
- Save the best performing models automatically

## Monitoring Training

You can monitor the training progress using TensorBoard:
```bash
tensorboard --logdir=./runs --load_fast=false
```

Then open your browser and navigate to `http://localhost:6006`
