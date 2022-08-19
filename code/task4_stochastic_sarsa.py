# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:01:38 2020

@author: vijay
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import policy


grid=[7,10]
num_states=grid[0]*grid[1]
#choose one of the following for 4 moves and king moves respectively
num_actions=4#for 4 moves
#actions=[Right,Left,Up,Down]

#num_actions=8#for king moves
#actions=[right,left,up,down,right-up,right-down,left-up,left-down]

start_state=[3,0]
end_state=[3,7]
windy_matrix=[0,0,0,1,1,1,2,2,1,0]

transition_reward=-1*np.ones((grid[0],grid[1],num_actions))
transition_reward[end_state[0],end_state[1]]=1
q_value=np.zeros((grid[0],grid[1],num_actions))

#environment defined 

time=8000
epsilon=0.1
alpha=0.5
episode_axis=np.zeros((time))
average_episode_axis=np.zeros((time))
randomSeed=50
for rseed in range(randomSeed):
    random.seed(rseed)    
    episode_axis=policy.stochastic_sarsa_4_8_moves(time,start_state,end_state,epsilon,q_value,windy_matrix,transition_reward,alpha)
    average_episode_axis+=episode_axis
average_episode_axis/=randomSeed
avg_episode_axis=average_episode_axis.tolist()
time_axis=list(range(1,time+1))

plt.plot(time_axis,avg_episode_axis)

plt.xlabel('no of steps')
plt.ylabel('no of episodes')
plt.title(f'Task4- Stochastic Windy Gridworld with {num_actions} moves')
plt.grid(True)
plt.savefig(f"../plots/stochastic-windy-world_{num_actions}_step_{randomSeed}_randomseed")
plt.show()
