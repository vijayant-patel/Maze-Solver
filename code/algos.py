# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 00:57:54 2020

@author: vijay
"""

import random 

def windy_nature(grid,i,j,k,windy_matrix):
    if k%2==0:
        if k==0:#right
            new_row=max(0,i-windy_matrix[j])
            new_column=min(grid[1]-1,j+1)
        elif k==2:#up
            new_row=max(i-windy_matrix[j]-1,0)
            new_column=j
        elif k==4:#right-up
            new_row=max(0,i-windy_matrix[j]-1)
            new_column=min(grid[1]-1,j+1)            
        else:#left-up
            new_row=max(0,i-windy_matrix[j]-1)
            new_column=max(0,j-1)
            
    else:
        if k==1:#left
            new_row=max(0,i-windy_matrix[j])
            new_column=max(0,j-1)
        elif k==3:#down
            new_row=max(0,i-windy_matrix[j]+1)
            new_row=min(new_row,grid[0]-1)
            new_column=j
        elif k==5:#right-down
            new_row=max(0,i-windy_matrix[j]+1)
            new_row=min(new_row,grid[0]-1)
            new_column=min(grid[1]-1,j+1)
        else:#left-down
            new_row=max(0,i-windy_matrix[j]+1)
            new_row=min(new_row,grid[0]-1)
            new_column=max(0,j-1)
            
    return [new_row,new_column]

def action(state,epsilon,q_value):
    num_actions=q_value.shape[2]
    p=(random.uniform(0,1.0))
    if p>epsilon:
        greedy=True
    else:
        greedy=False
    if greedy:
        state_list=q_value[state[0],state[1]].tolist()
        m=max(state_list)
        a=[]
        for i in range(num_actions):
            if state_list[i]==m:
                a.append(i)
        action_now=random.choice(a)
        action_now=state_list.index(m)
    else:
        action_now=random.randint(0,num_actions-1)
    return [action_now]

def stochastic_windy_nature(grid,i,j,k,wind_matrix):
    windy_matrix=wind_matrix.copy()
    if windy_matrix[j]!=0:
        p=random.choice([-1,0,1])
        if p==-1:
            windy_matrix[j]-=1
        elif p==1:
            windy_matrix[j]+=1   
    # from here same as normal windy_nature function
    if k%2==0:
        if k==0:#right
            new_row=max(0,i-windy_matrix[j])
            new_column=min(grid[1]-1,j+1)
        elif k==2:#up
            new_row=max(i-windy_matrix[j]-1,0)
            new_column=j
        elif k==4:#right-up
            new_row=max(0,i-windy_matrix[j]-1)
            new_column=min(grid[1]-1,j+1)            
        else:#left-up
            new_row=max(0,i-windy_matrix[j]-1)
            new_column=max(0,j-1)
            
    else:
        if k==1:#left
            new_row=max(0,i-windy_matrix[j])
            new_column=max(0,j-1)
        elif k==3:#down
            new_row=max(0,i-windy_matrix[j]+1)
            new_row=min(new_row,grid[0]-1)
            new_column=j
        elif k==5:#right-down
            new_row=max(0,i-windy_matrix[j]+1)
            new_row=min(new_row,grid[0]-1)
            new_column=min(grid[1]-1,j+1)
        else:#left-down
            new_row=max(0,i-windy_matrix[j]+1)
            new_row=min(new_row,grid[0]-1)
            new_column=max(0,j-1)
            
    return [new_row,new_column]
