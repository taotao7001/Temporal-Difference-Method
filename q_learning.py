#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:47:03 2020

@author: taotao
"""
from env import *
import matplotlib.pyplot as plt


#### Q-Learning ####
# hyperparameters
EPS = 0.01 # on policy and eps = 0.01
GAMMA = 0.9 
ALPHA = 1.0
ALPHA_MIN = 0.001
EPS_MIN = 0.01
MAX_STEP = 50  
TOTAL_STEPS = 1000000
ALPHA_DECAY = 0.9999954
EPS_DECAY = 1

# initiaise Q table
np.random.seed(0)
Q1 = np.random.rand(numS, numA)
Q2 = np.random.rand(numS, numA)

ERROR = [] 
s0 = 71
step = 0
while step < TOTAL_STEPS:
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
            a1_index = np.argmax(Q1[s])
            a1 = actions[a1_index]
            a2_index = np.argmax(Q2[s])
            a2 = actions[a2_index]

        a = [a1, a2] 
        r1 = R[s, 0]
        r2 = R[s, 1]

        if r1 != 0 or r2 != 0: 
            done = True

        # update Q
        q_s1 = Q1[s0,4]
        s_prime = transition(s, a) 
        Q1[s, a1_index] = Q1[s, a1_index] + ALPHA * ((1-GAMMA)*r1 + GAMMA * Q1[s_prime, :].max()*(1-done) - Q1[s, a1_index])
        Q2[s, a2_index] = Q2[s, a2_index] + ALPHA * ((1-GAMMA)*r2 + GAMMA * Q2[s_prime, :].max()*(1-done) - Q2[s, a2_index])
        # Update s
        s = s_prime
        step += 1
        ERROR.append(np.absolute(Q1[s0, 4] - q_s1))
        EPS = max(EPS*EPS_DECAY,EPS_MIN)
        ALPHA = max(ALPHA*ALPHA_DECAY,ALPHA_MIN)
        
        if step%200000 == 0:
            print(ALPHA)
            
            print(step)

        if done: break

plt.plot(ERROR)
plt.ylim([0,0.5])
plt.xlabel("Simulation Iteration")
plt.ylabel("Q-value Difference")
plt.show()


