# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:56:00 2021

@author: Longqiang

用Q-learning算法求解WindyWorld问题

"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

## 创建一个epsilon-贪婪策略
def create_egreedy_policy(env, Q, epsilon=0.1):
    # 内部函数
    def __policy__(state):
        NA = env.aspace_size
        A = np.ones(NA, dtype=float) * epsilon / NA  # 平均设置每个动作概率
        best = np.argmax(Q[state])                  # 选择最优动作
        A[best] += 1 - epsilon                      # 设定贪婪动作概率
        return A
    
    return __policy__  # 返回epsilon-贪婪策略函数
## Sarsa算法主程序
def sarsa(env, num_episodes=500, alpha=0.1, epsilon=0.1):
    NA = env.aspace_size
    # 初始化        
    Q = defaultdict(lambda: np.zeros(NA))           # 动作值
    egreedy_policy = create_egreedy_policy(env, Q, epsilon) # 贪婪策略函数
    rewards = []  # 记录每次episode的总奖励
    
    # 外层循环
    for _ in range(num_episodes):
        state = env.reset()                     # 环境状态初始化
        action_prob = egreedy_policy(state)     # 产生当前动作概率
        action = np.random.choice(np.arange(NA), p=action_prob)
        total_reward = 0  # 记录每个episode的总奖励
        
        # 内层循环
        while True:
            next_state, reward, end, info = env.step(action) # 交互一次
            total_reward += reward  # 累加奖励
            action_prob = egreedy_policy(next_state)         # 产生下一个动作
            next_action = np.random.choice(np.arange(NA), p=action_prob)
            Q[state][action] += alpha * (reward              # 策略评估
             + env.gamma * Q[next_state][next_action] - Q[state][action])
            
            # 到达终止状态退出本轮交互
            if end:                        
                break
                
            state = next_state      # 更新状态
            action = next_action    # 更新动作
        
        rewards.append(total_reward)  # 记录总奖励
    
    # 用表格表示最终策略
    P_table = np.ones((env.world_height, env.world_width)) * np.inf
    for state in env.get_sspace():
        P_table[state[0]][state[1]] = np.argmax(Q[state])
     
    # 返回最终策略、动作值和奖励记录
    return P_table, Q, rewards

## Q-learning主程序
def Qlearning(env, num_episodes=1000, alpha=0.1, epsilon=0.1):
    NA = env.aspace_size
    # 初始化    
    Q = defaultdict(lambda: np.zeros(NA))           # 动作值函数
    egreedy_policy = create_egreedy_policy(env, Q, epsilon) # 贪婪策略函数
    rewards = []  # 记录每次episode的总奖励
    
    # 外层循环
    for _ in range(num_episodes):
        state = env.reset()                     # 状态初始化
        total_reward = 0  # 记录每个episode的总奖励
                
        # 内层循环
        while True:
            action_prob = egreedy_policy(state) # 产生当前动作
            action = np.random.choice(np.arange(NA), p=action_prob)
            next_state, reward, end, info = env.step(action) # 交互一次
            total_reward += reward  # 累加奖励
            Q_max = np.max(Q[next_state])       # 最大动作值
            Q[state][action] += alpha * (       # 策略评估
                    reward + env.gamma * Q_max - Q[state][action])
            
            # 检查是否到达终止状态
            if end: 
                break                            
            
            state = next_state                  # 更新状态，进入下一次循环
        
        rewards.append(total_reward)  # 记录总奖励
    
    # 用表格表示最终策略
    P_table = np.ones((env.world_height, env.world_width)) * np.inf
    for state in env.get_sspace():
        P_table[state[0]][state[1]] = np.argmax(Q[state])
     
    # 返回最终策略、动作值和奖励记录
    return P_table, Q, rewards
def plot_policy(P_table, env, title="Policy"):
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(env.world_width))
    ax.set_yticks(np.arange(env.world_height))
    ax.set_xticklabels(np.arange(env.world_width))
    ax.set_yticklabels(np.arange(env.world_height))
    ax.grid(True)
    for i in range(env.world_height):
        for j in range(env.world_width):
            action = P_table[i, j]
            if action != np.inf:
                ax.text(j, i, action_symbols[action], ha='center', va='center', fontsize=18)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()

## 主程序
if __name__ == '__main__':
    # 构造WindyWorld环境
    import WindyWorld
    env = WindyWorld.WindyWorldEnv()
    import csv
    import time
    start = time.time()
    # 调用Q-learning算法
    P_table_Q, Q_Q, rewards_Q = Qlearning(env, num_episodes=1000, alpha=0.1, epsilon=0.1)
    stop = time.time()
    elapsed_time = stop - start
    # 输出Q-learning结果
    print('Q-learning P = ', P_table_Q)
    for state in env.get_sspace():
        print('{}: {}'.format(state, Q_Q[state]))
    csv_file = 'WindWorld.csv'
    try:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['4-4', f'{elapsed_time:.2f}'])
        print(f'时间已记录到 {csv_file}')
    except Exception as e:
        print(f'写入CSV文件时出错: {e}')
    
    # 调用SARSA算法
    start = time.time()
    P_table_S, Q_S, rewards_S = sarsa(env, num_episodes=1000, alpha=0.1, epsilon=0.1)
    stop = time.time()
    elapsed_time = stop - start
    # 输出SARSA结果
    print('SARSA P = ', P_table_S)
    for state in env.get_sspace():
        print('{}: {}'.format(state, Q_S[state]))
    try:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['4-3', f'{elapsed_time:.2f}'])
        print(f'时间已记录到 {csv_file}')
    except Exception as e:
        print(f'写入CSV文件时出错: {e}')

    # 绘制策略图
    plot_policy(P_table_Q, env, title="Q-learning Policy")
    plot_policy(P_table_S, env, title="SARSA Policy")

    # 绘制奖励变化情况对比图
    plt.plot(rewards_Q, label='Q-learning')
    plt.plot(rewards_S, label='SARSA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode in WindyWorld')
    plt.legend()
    plt.grid(True)
    plt.show()