#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:00:20 2020

@author: taotao
"""

from env import *
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


def uCEQ(s,Q1,Q2):

    Q1_flat = Q1[s].flatten()
    Q2_flat = Q2[s].flatten()
    
    c = -np.array(Q1_flat + Q2_flat, dtype="float")
    c = matrix(c)
    
    h = matrix(np.zeros(numA*(numA-1)*2+numA**2))
    
    b = matrix(1.)
    
    A = np.ones(numA**2)
    A = np.matrix(A, dtype='float')
    A = matrix(A)
    
    # Construct G table
    # row player
    zeros = np.zeros((1,5))
    for a in range(numA):
        diff = [Q1[s,a,i] - Q1[s,:,i] for i in range(numA)]
        diff_T = np.row_stack((diff)).T
        diff_T = np.delete(diff_T,a,axis=0)
    
        if a ==0:
            pi_p1 = np.row_stack((np.repeat(zeros,a*4, axis=0),diff_T ,np.repeat(zeros,(numA-a-1)*4, axis=0)))
    
        else:
            temp = np.row_stack((np.repeat(zeros,a*4, axis=0),diff_T ,np.repeat(zeros,(numA-a-1)*4, axis=0)))
            pi_p1 = np.column_stack((pi_p1,temp))
    pi_p1 = -pi_p1

    # column player
    for a in range(numA):
        diff =  [Q2[s,:,a] - Q2[s,:,i] for i in range(numA)]
        diff = np.row_stack((diff))
        diff= np.delete(diff,a,axis=0)
        if a ==0:
            pi_p2 = np.row_stack((np.repeat(zeros,a*4, axis=0),diff,np.repeat(zeros,(numA-a-1)*4, axis=0)))
    
        else:
            temp = np.row_stack((np.repeat(zeros,a*4, axis=0),diff ,np.repeat(zeros,(numA-a-1)*4, axis=0)))
            pi_p2 = np.column_stack((pi_p2,temp))
    pi_p2 = -pi_p2
    
    # non-negative constraints
    g_diag = np.zeros((numA**2,numA**2))
    np.fill_diagonal(g_diag,-1)
    
    # row stack matrixes
    g = np.row_stack((pi_p1,pi_p2,g_diag))
    G = matrix(g)
    solvers.options['show_progress'] = False
    solvers.options['feastol'] = 10e-5
    sol = solvers.lp(c, G, h, A, b,solver = None)
    sigmas = np.array(sol['x'].T)[0]
    sigmas -= sigmas.min() + 0.
    return sigmas.reshape((numA, numA)) / sigmas.sum(0)

#### Correlated - Q ####
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
        # solve linear programming
        try:
            sigma1 = uCEQ(s_prime,Q1,Q2)
            pi1[s_prime] = np.sum(np.array(sigma1).reshape((numA, numA)), axis=1)
            sigma2 = sigma1.T
            pi2[s_prime] = np.sum(np.array(sigma2).reshape((numA, numA)), axis=1)
        except:
            pass
        
        V1 = max(sum([pi1[s_prime, a] * Q1[s_prime,a,] for a in range(numA)])*(1-done) )
        V2 = max(sum([pi2[s_prime, a] * Q2[s_prime,a,] for a in range(numA)])*(1-done))

        Q1[s,a1_index,a2_index] =(1-ALPHA)* Q1[s,a1_index,a2_index] + ALPHA *((1-GAMMA)*r1+GAMMA * V1)
        Q2[s,a2_index,a1_index] =(1-ALPHA)* Q2[s,a2_index,a1_index] + ALPHA *((1-GAMMA)*r2+GAMMA * V2)
      
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