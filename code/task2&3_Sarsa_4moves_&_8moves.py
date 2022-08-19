# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 00:21:06 2020

@author: vijay
"""

import numpy as np
import algos
import random
import matplotlib.pyplot as plt
import policy


grid=[7,10]
num_states=grid[0]*grid[1]
#choose one of the following for 4 moves and king moves respectively

num_actions=4#for 4 moves
#actions=[Right,Left,Up,Down]

num_actions=8#for king moves
#actions=[right,left,up,down,right-up,right-down,left-up,left-down]

start_state=[3,0]
end_state=[3,7]
windy_matrix=[0,0,0,1,1,1,2,2,1,0]

next_state=np.zeros((grid[0],grid[1],num_actions,2))
transition_reward=-1*np.ones((grid[0],grid[1],num_actions))
transition_reward[end_state[0],end_state[1]]=1
q_value=np.zeros((grid[0],grid[1],num_actions))

for i in range(grid[0]):
    for j in range(grid[1]):
        if [i,j]!=start_state:    
            for k in range(num_actions):
                [next_state[i,j,k,0],next_state[i,j,k,1]]=algos.windy_nature(grid,i,j,k,windy_matrix)

#environment defined 

time=8000
epsilon=0.1
alpha=0.5
episode_axis=np.zeros((time))
average_episode_axis=np.zeros((time))
randomSeed=10
for rseed in range(randomSeed):
    random.seed(rseed)    
    episode_axis=policy.sarsa_4_8_moves(time,start_state,end_state,epsilon,q_value,next_state,transition_reward,alpha)
    average_episode_axis+=episode_axis
average_episode_axis/=randomSeed
avg_episode_axis=average_episode_axis.tolist()
time_axis=list(range(1,time+1))

plt.plot(time_axis,avg_episode_axis)

plt.xlabel('no of steps')
plt.ylabel('no of episodes')
plt.title(f'Task2- Windy Gridworld with {num_actions} moves')
plt.grid(True)
plt.savefig(f"../plots/windy-world_{num_actions}_step_{randomSeed}_randomseed")
plt.show()
