# Making A2C Work with LunarLander

This repository contains an implementation of the Advantage Actor-Critic (A2C) algorithm for training an agent to solve the LunarLander-v2 environment.

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
tensorboard --logdir runs
```

Then open your browser and navigate to `http://localhost:6006`
