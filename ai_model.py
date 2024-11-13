import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# Hyperparameters
STATE_SIZE = 15
ACTION_SIZE = 4  # Number of actions: bet, call, raise, fold
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.1  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay factor for epsilon
LEARNING_RATE = 0.001
MODEL_PATH = "poker_ai_model.pth"

# DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)

# Placeholder functions for resetting and stepping through the poker game
def reset_poker_game():
    initial_state = np.zeros(STATE_SIZE)
    done = False
    return initial_state, done

def step_poker_game(state, action):
    next_state = np.random.rand(STATE_SIZE)
    reward = compute_reward(state, action)  # Updated reward calculation
    done = np.random.choice([True, False], p=[0.1, 0.9])
    return next_state, reward, done

def compute_reward(state, action):
    """More realistic reward function"""
    if action == 0:  # Bet
        return 2  # Positive reward for aggressive betting
    elif action == 1:  # Call
        return 1  # Small reward for maintaining the game
    elif action == 2:  # Raise
        return 3  # Higher reward for aggressive play
    elif action == 3:  # Fold
        return -1  # Penalty for folding
    return 0

def select_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)  # Random action
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        return q_values.argmax().item()  # Action with highest Q-value

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_model(model):
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print("No existing model found. Starting fresh.")

def train_ai(episodes=1000, batch_size=64, replay_capacity=10000):
    dqn = DQN(STATE_SIZE, ACTION_SIZE)
    target_dqn = DQN(STATE_SIZE, ACTION_SIZE)
    
    load_model(dqn)
    target_dqn.load_state_dict(dqn.state_dict())
    
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(replay_capacity)

    epsilon = EPSILON_START

    for episode in range(episodes):
        state, done = reset_poker_game()
        total_reward = 0

        while not done:
            action = select_action(dqn, state, epsilon)
            next_state, reward, done = step_poker_game(state, action)

            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = batch

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_dqn(next_states).max(1)[0]
                targets = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)  # Decay epsilon
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    save_model(dqn)
    return dqn

# Add to the existing code in ai_model.py

def evaluate_ai(model, poker_game, num_games=100):
    """Evaluate the AI model's performance over a number of games"""
    poker_game.ai_model = model
    win_rate, avg_chips_won = poker_game.play_evaluation_games(num_games=num_games)
    print(f"Evaluation over {num_games} games:")
    print(f"AI Win Rate: {win_rate:.2f}%")
    print(f"Average Chips Won: {avg_chips_won:.2f}")

