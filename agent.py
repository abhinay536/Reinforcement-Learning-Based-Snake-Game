import random
import numpy as np
from collections import defaultdict
from game import SnakeGame, Direction, Point

# Q-learning parameters (final)
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01


class Agent:
    def __init__(self):
        self.epsilon = EPSILON
        self.q_table = defaultdict(lambda: np.zeros(3))

    def get_state(self, game):
        head = game.head

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = (
            # Danger straight
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # Danger right
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            # Danger left
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),

            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        )

        return tuple(int(x) for x in state)

    def get_action(self, state):
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            move = np.argmax(self.q_table[state])

        action = [0, 0, 0]
        action[move] = 1
        return action

    def update_q(self, state, action, reward, next_state):
        action_index = np.argmax(action)

        old_value = self.q_table[state][action_index]
        next_max = np.max(self.q_table[next_state])

        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        self.q_table[state][action_index] = new_value

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

from helper import plot

def train():
    agent = Agent()
    game = SnakeGame()

    scores = []
    mean_scores = []
    total_score = 0

    n_games = 400

    for i in range(n_games):
        game.reset()

        while True:
            state = agent.get_state(game)
            action = agent.get_action(state)

            game_over, score = game.play_step(action)

            if game_over:
                reward = -10
            elif game.head == game.food:
                reward = 10
            else:
                reward = -0.1

            next_state = agent.get_state(game)

            agent.update_q(state, action, reward, next_state)

            if game_over:
                agent.decay_epsilon()

                scores.append(score)
                total_score += score
                mean_score = total_score / (i + 1)
                mean_scores.append(mean_score)

                print(f"Game {i+1}, Score: {score}, Mean Score: {mean_score:.2f}")

                plot(scores, mean_scores)
                break

    print("Training finished!")

if __name__ == "__main__":
    train()