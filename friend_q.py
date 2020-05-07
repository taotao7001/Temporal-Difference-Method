#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:54:42 2020

@author: taotao
"""

from env import *
import matplotlib.pyplot as plt


#### Friend - Q ####
# hyperparameters
EPS = 1.0 # off policy
GAMMA = 0.9 
ALPHA = 1
ALPHA_MIN = 0.001
EPS_MIN = 1
MAX_STEP = 50  
TOTAL_ITS = 50000
ALPHA_DECAY =  0.9999
EPS_DECAY = 1

# initiailse tables
np.random.seed(1)
Q1 = np.random.random((numS,numA,numA))
Q2 = np.random.random((numS,numA,numA))

ERROR = [] 
s0 = 71
step = 0
while step < TOTAL_ITS:
    
    # reset for each game
    s = s0  
    done = False
    # stop if excceed maximum steps
    for t in range(MAX_STEP):
        if np.random.uniform() <= EPS:
            a1_index = np.random.randint(5)
            a1 = actions[a1_index]
            a2_index = np.random.randint(5)
            a2 = actions[a2_index]
        
        else:
            joint_index1 = np.argmax([Q1[s]], axis=None)
            joint_index2 = np.argmax([Q2[s]], axis=None)
            a1_index,_ = np.unravel_index(joint_index1, Q1[s].shape)
            a2_index,_ = np.unravel_index(joint_index2, Q2[s].shape)
            
            a1= actions[a1_index]
            a2= actions[a2_index]
        
        a = [a1, a2] 
       
        r1 = R[s, 0]
        r2 = R[s, 1]

        if r1 != 0 or r2 != 0: 
            done = True

        # update Q, SOUTH INDEX:4, STICK INDEX:2
        q_s1 = Q1[s0,4,2]
        s_prime = transition(s,a) 
        Nash_Q1 = Q1[s_prime,:,:].max()*(1-done)
        Nash_Q2 = Q2[s_prime,:,:].max()*(1-done)
        Q1[s,a1_index,a2_index] =(1-ALPHA)* Q1[s,a1_index,a2_index] + ALPHA *((1-GAMMA)*r1+GAMMA * Nash_Q1)
        Q2[s,a2_index,a1_index] =(1-ALPHA)* Q2[s,a2_index,a1_index] + ALPHA *((1-GAMMA)*r2+GAMMA * Nash_Q2)
      
        # Update s
        s = s_prime
        step+= 1
        ERROR.append(np.absolute(Q1[s0,4,2] - q_s1))
        EPS = max(EPS*EPS_DECAY,EPS_MIN)
        ALPHA = max(ALPHA*ALPHA_DECAY,ALPHA_MIN)
 

        if done: break
plt.plot(ERROR)
plt.ylim([0,0.5])
plt.xlabel("Simulation Iteration")
plt.ylabel("Q-value Difference")
plt.show()