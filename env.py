# Packages
import numpy as np

# Initialisation
numS = 112 # 8*7*2 = 112, 
# State Matrix: possesseion_p1,p1,p2
s1 = [s for s in range(8)]
s2 = [s for s in range(8)]
# Two players can't be in same state, 0 if p1 has the ball
states1 = [[0,i,j] for i in s1 for j in s2 if i!=j]
states2 = [[1,i,j] for i in s1 for j in s2 if i!=j]
S = states1+ states2
S = np.array(S) 


# Reward matrix: r1,r2
# p1 + 100 if has ball and in state 0 or 4, p2 has ball and in state 0 or 4
r1 = np.zeros((numS,1))
r2 = np.zeros((numS,1))
p1_win = [ind for ind,[i,j,k] in enumerate(S) if (i==0 and (j==0 or j==4)) or (i==1 and (k==0 or k==4))]
p1_lose = [ind for ind,[i,j,k] in enumerate(S) if (i==1 and (k==3 or k==7)) or (i==0 and (j==3 or j==7))]
r1[p1_win] = 100
r1[p1_lose] = -100
r2[p1_win] = -100
r2[p1_lose] = 100
R = np.column_stack((r1,r2))


# Action matrix, a1,a2
# N -4,S +4,E +1,W -1,STICK 0
numA = 5
actions = [-4,-1,0,1,4]
A = [[i,j] for i in actions for j in actions]
A = np.array(A)

def transition(s,a):
    p1_poss,s1,s2 = S[s]
    # equivalent to stick if outside the pitch
    if s1 + a[0] >7 or s1+ a[0] <0:
        a[0] = 0
    if s2 + a[1] >7 or s2 + a[1] <0:
        a[1] = 0
        
    s1_temp = s1 + a[0]
    s2_temp = s2 + a[1]
    
    s_ind = s
    
    # collision
    if s1_temp == s2_temp:
        first = np.random.randint(2)
        if first==0: # p1 move first          
            # p1 has ball, p1 goes to empty cell, B follows but can't and can't steal the ball
            # p2 has ball, p1 goes to empty cell, B follows,lose the ball and can't move
            s_next_temp = [0,s1_temp,s2]
        else: # p2 move first
            s_next_temp = [1,s1,s2_temp]
    else:
        s_next_temp = [p1_poss,s1_temp,s2_temp]
    
    s_next = np.array(s_next_temp)
    # Get next state index
    for i in range(numS):
        if np.array_equal(S[i],s_next):
            s_ind = i
            
    return s_ind

