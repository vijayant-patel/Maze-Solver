# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 03:48:36 2020

@author: vijay
"""
import algos
import numpy as np

def sarsa_4_8_moves(time,start_state,end_state,epsilon,q_values,next_state,transition_reward,alpha):
    t=0
    q_value=q_values.copy()
    episode_axis=np.zeros((time))
    num_episodes=0
    while (t<time):
        num_episodes+=1
        state=start_state
        [action_now]=algos.action(state,epsilon,q_value)
        while state!=end_state:

            state_next=next_state[state[0],state[1],action_now].tolist()
            state_next[0]=int(state_next[0])
            state_next[1]=int(state_next[1])
            [action_next]=algos.action(state_next,epsilon,q_value)
            target=transition_reward[state_next[0],state_next[1],action_next]+q_value[state_next[0],state_next[1],action_next]
            q_value[state[0],state[1],action_now]*=(1-alpha)
            q_value[state[0],state[1],action_now]+=alpha*target
            
            # updating
            episode_axis[t]=num_episodes
            state=state_next
            t+=1
            action_now=action_next
            if t==time:
                break
    return episode_axis

def update_function(function,e,num_episodes,start_state,end_state,epsilon,q_values,next_state,transition_reward,alpha):
    q_value=q_values.copy()
    num_actions=q_value.shape[2]
    t=0
    n=0
    time_new_episode=np.zeros((num_episodes))
    while (n<num_episodes):
        n+=1
        state=start_state
        [action_now]=algos.action(state,epsilon,q_value)
        while state!=end_state:

            state_next=next_state[state[0],state[1],action_now].tolist()
            state_next[0]=int(state_next[0])
            state_next[1]=int(state_next[1])
            [action_next]=algos.action(state_next,epsilon,q_value)
            
            if function=='Sarsa':
                target=transition_reward[state_next[0],state_next[1],action_next]+q_value[state_next[0],state_next[1],action_next]
            elif function=='Q-learning':
                target=transition_reward[state_next[0],state_next[1],action_next]+max(q_value[state_next[0],state_next[1]])
            else:
                target=transition_reward[state_next[0],state_next[1],action_next]+(1-e)*max(q_value[state_next[0],state_next[1]])
                sum=0
                for i in range(num_actions):
                    sum+=q_value[state_next[0],state_next[1],i]
                target+=sum*e/num_actions
            
            q_value[state[0],state[1],action_now]*=(1-alpha)
            q_value[state[0],state[1],action_now]+=alpha*target
            
            # updating
 
            state=state_next
            t+=1
            action_now=action_next
            if n==num_episodes:
                break
        time_new_episode[n-1]=t
    return time_new_episode

def stochastic_sarsa_4_8_moves(time,start_state,end_state,epsilon,q_values,windy_matrix,transition_reward,alpha):
    t=0
    q_value=q_values.copy()
    grid=[q_value.shape[0],q_value.shape[1]]
    episode_axis=np.zeros((time))
    num_episodes=0
    while (t<time):
        num_episodes+=1
        state=start_state
        [action_now]=algos.action(state,epsilon,q_value)
        while state!=end_state:

            state_next=algos.stochastic_windy_nature(grid,state[0],state[1],action_now,windy_matrix)
            state_next[0]=int(state_next[0])
            state_next[1]=int(state_next[1])
            [action_next]=algos.action(state_next,epsilon,q_value)
            target=transition_reward[state_next[0],state_next[1],action_next]+q_value[state_next[0],state_next[1],action_next]
            q_value[state[0],state[1],action_now]*=(1-alpha)
            q_value[state[0],state[1],action_now]+=alpha*target
            
            # updating
            episode_axis[t]=num_episodes
            state=state_next
            t+=1
            action_now=action_next
            if t==time:
                break
    return episode_axis