# Reinforcement Learning Snake Agent using Q-Learning 🐍🤖

## Overview
This project implements a Snake game where an AI agent learns to play using Q-learning.
The agent improves over time by interacting with the environment and maximizing rewards.

## Objective
Train an agent that learns to:
- Avoid collisions
- Reach food efficiently
- Maximize score without hardcoded rules

## Reinforcement Learning Setup

Environment:
Custom Snake game built using Pygame

State:
- Danger straight
- Danger right
- Danger left
- Current direction (left/right/up/down)
- Food location (left/right/up/down)

Actions:
- [1, 0, 0] → Move straight
- [0, 1, 0] → Turn right
- [0, 0, 1] → Turn left

Reward:
- +10 → Eat food
- -10 → Collision
- -0.1 → Each step

Algorithm:
Q-Learning

Q(s, a) = Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]

Exploration:
Epsilon-greedy strategy with decay

## Results
The agent improves gradually over time.

(Add your graph screenshot here later)

## Tech Stack
- Python 3.12
- Pygame
- NumPy
- Matplotlib

## Project Structure
reinforcement-learning-snake-qlearning/
│
├── game.py
├── agent.py
├── helper.py
├── README.md

## How to Run

1. Install dependencies:
pip install pygame numpy matplotlib

2. Run:
python agent.py

## Observations
- Initial scores are low due to exploration
- Performance improves over time
- Mean score increases gradually

## Future Improvements
- Deep Q-Learning (DQN)
- Save trained model
- Better state representation

