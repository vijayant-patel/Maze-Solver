# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 00:21:06 2020

@author: vijay
"""

import numpy as np
import algos
import random
import policy
import matplotlib.pyplot as plt



grid=[7,10]
num_states=grid[0]*grid[1]

num_actions=4#for 4 moves
#actions=[Right,Left,Up,Down]

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
function_list=['Sarsa','Q-learning','Expected Sarsa']
num_episodes=200
epsilon=0.1
alpha=0.5
episode_axis=list(range(num_episodes+1))
e=0.1
randomSeed=10
for function in function_list:
    average_time_axis=np.zeros((num_episodes)) 
    for rseed in range(randomSeed):
        random.seed(rseed)    
        time_axis=policy.update_function(function,e,num_episodes,start_state,end_state,epsilon,q_value,next_state,transition_reward,alpha)
        average_time_axis+=time_axis
    average_time_axis/=randomSeed
    avg_time_axis=average_time_axis.tolist()
    avg_time_axis=[0]+avg_time_axis
    plt.plot(avg_time_axis,episode_axis,label=function)

plt.xlim([0,9000])
plt.xticks(range(0,9000,1000))
plt.xlabel("no. of steps")
plt.ylabel("no. of episodes")
plt.title("Task-5 Windy Gridworld algorithm comparision")
plt.grid(True)
plt.legend()
plt.savefig(f"../plots/Task-5 windy-gridworld_algoritm_comparision_{randomSeed}_randomseed")    
plt.show()
            
        
        
        
        
        
