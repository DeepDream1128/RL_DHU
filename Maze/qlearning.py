import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import imageio

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_size = 6
        self.observation_space = gym.spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = gym.spaces.Discrete(4)  # 上下左右四个动作

        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size

        self.obstacles = [(1, 2), (3, 3), (3, 5), (5, 4)]
        self.goal = (5, 5)
        self.intermediate_reward_pos = (2, 4)
        self.intermediate_reward_collected = False

        self.agent_pos = [0, 0]

        self.screen = None
        self.clock = None
        self.font = None

        self.current_episode = 0
        self.current_step = 0
        self.total_reward = 0

        self.visited_cells = np.zeros((self.grid_size, self.grid_size), dtype=int)
        

    def reset(self):
        self.agent_pos = [0, 0]
        self.intermediate_reward_collected = False
        self.current_step = 0
        self.total_reward = 0
        self.visited_cells = np.zeros((self.grid_size, self.grid_size), dtype=int)  # 重置路径记录
        self.visited_cells[self.agent_pos[0], self.agent_pos[1]] += 1
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        reward = -1  # 每走一步有-1的惩罚

        if action == 0:  # 上
            next_pos = [self.agent_pos[0] - 1, self.agent_pos[1]]
        elif action == 1:  # 下
            next_pos = [self.agent_pos[0] + 1, self.agent_pos[1]]
        elif action == 2:  # 左
            next_pos = [self.agent_pos[0], self.agent_pos[1] - 1]
        elif action == 3:  # 右
            next_pos = [self.agent_pos[0], self.agent_pos[1] + 1]

        if self._is_valid_position(next_pos):
            self.agent_pos = next_pos
            self.visited_cells[self.agent_pos[0], self.agent_pos[1]] += 1  # 记录经过的路径
        else:
            reward -= 15  # 尝试进入障碍物时的惩罚

        if tuple(self.agent_pos) == self.intermediate_reward_pos and not self.intermediate_reward_collected:
            reward += 5
            self.intermediate_reward_collected = True

        if tuple(self.agent_pos) == self.goal:
            reward += 20
            done = True
        else:
            done = False

        self.total_reward += reward
        return self._get_state(), reward, done, {}

    def _is_valid_position(self, pos):
        if pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size:
            return False
        if tuple(pos) in self.obstacles:
            return False
        return True

    def _get_state(self):
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

    def render(self, mode='human', save_frame=False, frames=[]):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)

        self.screen.fill((255, 255, 255))

        # Draw grid
        for x in range(0, self.window_size, self.cell_size):
            for y in range(0, self.window_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            self._draw_cell(obstacle, (0, 0, 0))

        # Draw goal
        self._draw_cell(self.goal, (0, 255, 0))

        # Draw intermediate reward if not collected
        if not self.intermediate_reward_collected:
            self._draw_cell(self.intermediate_reward_pos, (0, 0, 255))

        # Draw visited cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.visited_cells[i, j] > 0:
                    intensity = min(255, self.visited_cells[i, j] * 30)
                    self._draw_cell((i, j), (255 - intensity, 255 - intensity, 255 - intensity), 1)

        # Draw agent
        self._draw_cell(tuple(self.agent_pos), (255, 0, 0))

        # Draw text
        self._draw_text(f'Episode: {self.current_episode}', 10, 10)
        self._draw_text(f'Step: {self.current_step}', 10, 40)
        self._draw_text(f'Reward: {self.total_reward}', 10, 70)

        pygame.display.flip()
        self.clock.tick(30)
        
        if save_frame:
            frame = pygame.surfarray.array3d(self.screen)
            frames.append(np.transpose(frame, (1, 0, 2)))

    def _draw_cell(self, pos, color, thickness=0):
        rect = pygame.Rect(pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect)
        if thickness > 0:
            pygame.draw.rect(self.screen, (0, 0, 0), rect, thickness)
    
    def _draw_text(self, text, x, y):
        img = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(img, (x, y))
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
          

env = GridWorldEnv()
env.render()

import random
import time

def q_learning(env, num_episodes=50, alpha=0.1, gamma=0.99, epsilon=0.1, render_interval=10, save_video=False, video_path="q_learning.mp4"):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    frames = []
    
    for episode in range(num_episodes):
        state = env.reset()
        env.current_episode = episode
        total_reward = 0
        done = False

        step_count = 0
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action]
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            state = next_state
            total_reward += reward
            step_count += 1

            if step_count > 50:  # 防止无限循环
                print(f'Episode {episode} exceeded 50 steps, terminating early.')
                break
            
            if episode % render_interval == 0:
                env.render(save_frame=save_video, frames=frames)
                time.sleep(0.1)

        rewards.append(total_reward)
        print(f'Episode {episode}: Total Reward: {total_reward}, Steps: {step_count}')
        
        # Render at the end of each episode
        if episode % render_interval == 0:
            env.render()
            time.sleep(0.1)
    if save_video:
        imageio.mimsave(video_path, frames, fps=10)
    return q_table, rewards

# 超参数
alpha = 0.5
gamma = 0.995
epsilon = 0.01
num_episodes = 50
render_interval = 10  # 每50个episode渲染一次

q_table, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon, render_interval, save_video=True, video_path="q_learning.mp4")

import time

def play(env, q_table):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    env.current_episode = 50
    frames=[]
    while not done:
        env.current_step = step_count
        env.render(save_frame=True, frames=frames)
        time.sleep(0.5)
        
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
    
    env.render(save_frame=True, frames=frames)
    if len(frames)>0:
        imageio.mimsave("q_learning_play.mp4", frames, fps=5)
    print(f"Total reward: {total_reward}")

play(env, q_table)
env.close()
