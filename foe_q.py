#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:58:34 2020

@author: taotao
"""

from env import *
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def minmax(s,Q):
    '''
    Input: Q table, state
    '''
    c = matrix(np.append(np.zeros(numA),np.array([-1])))
    h = matrix(np.zeros(numA*2))
    b = matrix(1.)
    A = np.append(np.ones(numA),0 )
    A = np.matrix(A, dtype='float')
    A = matrix(A)
    # construct G table by stacking constraints and non-zero conditions
    g_diag = np.zeros((numA,numA))
    np.fill_diagonal(g_diag,-1)
    g_q = -Q[s].T
    g = np.row_stack((g_q,g_diag))
    # u column
    u = np.append(np.ones(numA),np.zeros(numA)).T
    G = matrix(np.column_stack((g,u)))
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, A, b,solver = 'glpk')
    sigmas = np.array(sol['x'].H[0:5])
    sigmas = np.concatenate((sigmas), axis=None)
    sigmas -= sigmas.min() + 0.
    return sigmas / sigmas.sum(0)

#### Foe - Q ####
# hyperparameters
EPS = 1.0 # off policy
GAMMA = 0.9 
ALPHA = 1
ALPHA_MIN = 0.001
EPS_MIN = 1
MAX_STEP = 50  
TOTAL_ITS = 50000
ALPHA_DECAY = 0.9999
EPS_DECAY = 1

# initiailse tables
np.random.seed(1)
Q1 = np.random.random((numS,numA,numA))
Q2 = np.random.random((numS,numA,numA))
pi1 = np.full((numS,numA),1/numA)
pi2 = np.full((numS,numA),1/numA)

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
            a1_index = np.random.choice(range(numA), p=pi1[s])
            a2_index = np.random.choice(range(numA), p=pi2[s])
   
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
        # solve for adversial eq
        pi1[s_prime] = minmax(s_prime,Q1)
        pi2[s_prime] = minmax(s_prime,Q2)
        Nash_Q1 = max(sum([pi1[s_prime, a] * Q1[s_prime,a, ] for a in range(numA)])*(1-done)) 
        Nash_Q2 = max(sum([pi2[s_prime, a] * Q2[s_prime,a, ] for a in range(numA)])*(1-done))

        Q1[s,a1_index,a2_index] =(1-ALPHA)* Q1[s,a1_index,a2_index] + ALPHA *((1-GAMMA)*r1+GAMMA * Nash_Q1)
        Q2[s,a2_index,a1_index] =(1-ALPHA)* Q2[s,a2_index,a1_index] + ALPHA *((1-GAMMA)*r2+GAMMA * Nash_Q2)
      
        # Update s
        s = s_prime
        step+= 1
#         if step % 10000 ==0:
#             print(step)
#             print(ALPHA)
#             print(EPS)
        ERROR.append(np.absolute(Q1[s0,4,2] - q_s1))
        EPS = max(EPS*EPS_DECAY,EPS_MIN)
        ALPHA = max(ALPHA*ALPHA_DECAY,ALPHA_MIN)
 

        if done: break
plt.plot(ERROR)
plt.ylim([0,0.5])
plt.xlabel("Simulation Iteration")
plt.ylabel("Q-value Difference")
plt.show()