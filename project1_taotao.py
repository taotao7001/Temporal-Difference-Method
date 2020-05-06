#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:43:04 2020

@author: taotao
"""

import numpy as np
import matplotlib.pyplot as plt
import random

#### Initialisation ####
true_w = np.array([1/6,1/3,1/2,2/3,5/6])
x1 = np.array([1,0,0,0,0])
x2 = np.array([0,1,0,0,0])
x3 = np.array([0,0,1,0,0])
x4 = np.array([0,0,0,1,0])
x5 = np.array([0,0,0,0,1])
x = [x1,x2,x3,x4,x5]

#### Generation of random walks ####
random.seed(666)
num = 1000
sequences = [[] for i in range(num)]
seq = 0
while seq <= 999:
    k = 3
    while 0<k<6:
        sequences[seq].append(k)
        if random.random() > 0.5: k += 1
        else: k -= 1            
    if sequences[seq][-1] ==5:
        sequences[seq].append(6)
    else:
        sequences[seq].append(0)
#     if len(sequences[seq]) > 14: # comment this line to remove limit on sequence length
#         sequences[seq] = [] # comment this line to remove limit on sequence length
#         seq -= 1 # comment this line to remove limit on sequence length
    
    seq += 1
    
#### Figure 3 ####
alpha = 0.01
lamda = np.array([0,0.1,0.3,0.5,0.7,0.9,1])

rmse_lamda = np.zeros(7)
for n in range(len(lamda)):
    rmse = []
    lam = lamda[n]
    print(lam)
    for i in range(0,100):
        # For each training set, repeat until no signigicant change
        sequence = sequences[i*10:(i*10+10)]
        # initialise w for training set
        w = np.array([random.random() for i in range(5)])
        epsilon = 1
        # continue until convergence
        while epsilon > 10**(-6):
            # Accumulate delta_W until 10th finishes
            delta_w = np.zeros(5)
            for seq in sequence:
                m = len(seq)-1
                z = int(seq[-1] == 6)
                for t in range(1,m+1):
                    sumGrad = np.zeros(5)
                    for k in range(1,t+1):
                        grad = np.power(lam,t-k) * x[seq[k-1]-1]
                        sumGrad += grad
                    if t == m:
                        delta_wt = alpha * (z-w[seq[t-1]-1]) * sumGrad   
                    else:
                        delta_wt = alpha * (w[seq[t]-1]-w[seq[t-1]-1]) * sumGrad
                    delta_w += delta_wt
            # After 10 sequences, update w vector
            w += delta_w 
            epsilon = np.amax(np.absolute(delta_w ))
        # Calculate RMSE for each training set
        rmse.append(np.sqrt(np.sum(np.power(true_w-w,2))/5))  
    rmse_lamda[n] = np.average(rmse)

plt.figure(figsize=(10,8))
plt.plot(lamda,rmse_lamda,'-o')
plt.ylabel('ERROR',size=24)
plt.xlabel('λ',size=24)
plt.annotate('Widrow-Hoff',(0.75, rmse_lamda[-1]),fontsize=20)
plt.show()


#### Figure 4 ####
lamda = np.arange(0.0, 1.1, 0.1)
alphas = np.arange(0.0, 0.65, 0.05)
errors = np.zeros([len(lamda),len(alphas)])

for n in range(len(lamda)):
    lam = lamda[n]
    for a in range(len(alphas)):
        alpha = alphas[a]  
        rmse = []
        for i in range(100):
            # initialise w with all components set to 0.5
            w = np.array([0.5,0.5,0.5,0.5,0.5])
            sequence = sequences[i*10:(i*10+10)]  
            for seq in sequence:
                m = len(seq)-1
                z = int(seq[-1] == 6)
                # sum delta_wt from state 1 to state termination
                delta_wt_sum = np.zeros(5)
                for t in range(1,m+1):
                    sumGrad = np.zeros(5)
                    for k in range(1,t+1):
                        grad = np.power(lam,t-k) * x[seq[k-1]-1]
                        sumGrad += grad 
                    if t == m:
                        delta_wt = alpha * (z-w[seq[t-1]-1]) * sumGrad   
                    else:
                        delta_wt = alpha * (w[seq[t]-1]-w[seq[t-1]-1]) * sumGrad
                    delta_wt_sum += delta_wt
                # Update w vector after each sequence
                w = w + delta_wt_sum
                
            # Calculate RMSE for each training set
            rmse.append(np.sqrt(np.sum(np.power(true_w-w,2))/5) ) 
         
        errors[n][a] = np.average(rmse)

plt.figure(figsize=(10,8))
plt.plot(alphas,errors[0],'-o',label='λ = {}'.format(lamda[0]))
plt.plot(alphas,errors[3],'-o',label='λ = {}'.format(round(lamda[3],2)))
plt.plot(alphas,errors[8],'-o',label='λ = {}'.format(lamda[8]))
plt.plot(alphas,errors[10],'-o',label='λ = {} (Widrow-Hoff)'.format(lamda[10],1))
plt.ylabel('ERROR',size=20)
plt.xlabel('α',size=20)
plt.legend()
plt.show() 


#### Figure 5 ####
best_error = np.min(errors,axis=1)
plt.figure(figsize=(10,8))
plt.plot(lamda,best_error,'-o')
plt.ylabel('ERROR USING BEST α',size=24)
plt.xlabel('λ',size=24)
plt.annotate('Widrow-Hoff',(0.75,best_error[-1]),fontsize=20)
plt.show()

