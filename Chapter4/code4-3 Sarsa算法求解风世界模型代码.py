# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:56:00 2021

@author: Longqiang

用sarsa算法求解WindyWorld问题

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

## 主程序
if __name__ == '__main__':
    # 构造WindyWorld环境
    import WindyWorld
    env = WindyWorld.WindyWorldEnv()
    import csv
    import time
    start = time.time()
    # 调用Sarsa算法
    P_table, Q, rewards = sarsa(env, num_episodes=1000, alpha=0.1, epsilon=0.1)
    stop = time.time()
    elapsed_time = stop - start
    # 输出   
    print('P = ', P_table)
    for state in env.get_sspace():
        print('{}: {}'.format(state, Q[state]))
    csv_file = 'WindWorld.csv'
    try:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['4-3', f'{elapsed_time:.2f}'])
        print(f'时间已记录到 {csv_file}')
    except Exception as e:
        print(f'写入CSV文件时出错: {e}')
    
    # 绘制奖励变化情况
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode in WindyWorld')
    plt.show()
