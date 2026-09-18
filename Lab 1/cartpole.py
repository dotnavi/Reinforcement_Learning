"""
Reinforcement Learning Lab: Gymnasium Basics with CartPole-v1
Tasks:
  1. Environment Setup & Verification
  2. Create & Initialize an RL Environment
  3. Explore Observation and Action Spaces
  4. Execute a Random Agent for One Complete Episode
"""

import subprocess
import sys

# Task 1: Environment Setup
# Ensure necessary dependencies are installed
required_packages = ["gymnasium", "numpy", "matplotlib"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("TASK 1: ENVIRONMENT SETUP & VERIFICATION")
print("=" * 60)
print(f"Gymnasium version: {gym.__version__}")
print(f"NumPy version:     {np.__version__}")
print("Dependencies verified successfully!\n")


# Task 2: Create and Initialize an RL Environment
print("=" * 60)
print("TASK 2: CREATE & INITIALIZE CARTPOLE-V1")
print("=" * 60)
env = gym.make("CartPole-v1")

# Reset returns (initial_observation, info_dict)
initial_observation, info = env.reset(seed=42)

print("Environment: CartPole-v1")
print(f"Initial Observation: {initial_observation}")
print(f"Initial Info:        {info}\n")


# Task 3: Explore Observation and Action Spaces
print("=" * 60)
print("TASK 3: EXPLORE OBSERVATION & ACTION SPACES")
print("=" * 60)
print(f"Observation Space:          {env.observation_space}")
print(f"Observation Space Type:     {type(env.observation_space).__name__}")
print(f"Observation High Limits:    {np.round(env.observation_space.high, 4)}")
print(f"Observation Low Limits:     {np.round(env.observation_space.low, 4)}")
print(f"Action Space:               {env.action_space}")
print(f"Action Space Type:          {type(env.action_space).__name__}")
print(f"Number of Possible Actions: {env.action_space.n}\n")


# Task 4: Execute a Random Agent
print("=" * 60)
print("TASK 4: EXECUTE RANDOM AGENT (1 EPISODE)")
print("=" * 60)

obs, info = env.reset(seed=42)
total_reward = 0.0
step_count = 0
terminated = False
truncated = False

print(f"{'Step':<6} | {'Action':<8} | {'Reward':<8} | {'Terminated':<12} | {'Truncated':<12} | Observation [x, x_dot, theta, theta_dot]")
print("-" * 95)

while not (terminated or truncated):
    # Select random action from action space
    action = env.action_space.sample()

    # Step environment
    next_obs, reward, terminated, truncated, info = env.step(action)

    step_count += 1
    total_reward += reward

    obs_str = str(np.round(next_obs, 4))
    print(f"{step_count:<6} | {action:<8} | {reward:<8.1f} | {str(terminated):<12} | {str(truncated):<12} | {obs_str}")

    obs = next_obs

env.close()

print("-" * 95)
print(f"Total Steps Taken:  {step_count}")
print(f"Cumulative Reward:  {total_reward}")
print("=" * 60)