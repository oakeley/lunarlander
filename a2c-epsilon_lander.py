import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os

# Define the Actor-Critic Network architecture
class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        shared_output = self.shared(x)
        return self.actor(shared_output), self.critic(shared_output)

# A2C Agent
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.gamma = 0.99
        self.action_size = action_size
        
        # For tracking performance
        self.episode_rewards = []
        self.running_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_probs[0]
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get values for current and next states
        _, current_values = self.model(states)
        _, next_values = self.model(next_states)
        current_values = current_values.squeeze()
        next_values = next_values.squeeze()
        
        # Compute returns and advantages
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - current_values
        
        # Get action probabilities and values
        action_probs, values = self.model(states)
        values = values.squeeze()
        
        # Compute action log probabilities
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

    def plot_training_progress(self):
        plt.figure(figsize=(15, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # Plot running average of rewards
        plt.subplot(2, 2, 2)
        plt.plot(self.running_rewards)
        plt.title('Running Average Reward (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        
        # Plot actor losses
        plt.subplot(2, 2, 3)
        plt.plot(self.actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        # Plot critic losses
        plt.subplot(2, 2, 4)
        plt.plot(self.critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

def train_agent():
    # Install required packages if not present
    import subprocess
    import sys
    try:
        import moviepy
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
        
    # Set up video directory
    video_dir = '/home/edward/Videos/'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        
    # Create and wrap environment
    env = gym.make('LunarLander-v2')
    env = gym.wrappers.Monitor(env, video_dir, force=True, video_callable=lambda episode_id: True)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = A2CAgent(state_size, action_size)
    episodes = 1000
    update_frequency = 5  # Number of steps before updating the model
    
    running_reward = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        while not done:
            action, _ = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            total_reward += reward
            
            # Update the model every update_frequency steps or at the end of episode
            if len(states) == update_frequency or done:
                actor_loss, critic_loss = agent.update(states, actions, rewards, next_states, dones)
                agent.actor_losses.append(actor_loss)
                agent.critic_losses.append(critic_loss)
                states, actions, rewards, next_states, dones = [], [], [], [], []
        
        # Store episode results
        agent.episode_rewards.append(total_reward)
        running_reward = 0.05 * total_reward + 0.95 * running_reward
        agent.running_rewards.append(running_reward)
        
        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Running Reward: {running_reward:.2f}")
        
        # Plot training progress every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.plot_training_progress()
        
        # Early stopping if solved
        if running_reward >= 200:
            print(f"Environment solved in {episode + 1} episodes!")
            agent.plot_training_progress()
            break
    
    env.close()
    return agent

if __name__ == "__main__":
    trained_agent = train_agent()
