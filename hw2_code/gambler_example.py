import numpy as np
from dynamicProgramming import *
import matplotlib.pyplot as plt

# Set up P and R
def get_P_and_R_gambler(prob_head):
    # initialization    
    P = np.zeros([101,100,101])
    R = np.zeros([101,100,101])

    for i in range(100):
        R[i,:,100] = 1
        for j in range(i+1):
            if j != 0:
                if i+j<101:
                    P[i,j,i+j] = prob_head
                else:
                    P[i, j, 100] = prob_head

                if i-j >= 0:
                    P[i,j,i-j] = 1-prob_head
                else:
                    P[i, j, 0] = 1 - prob_head

            elif j == 0:
                P[i, j, i] = 1

    return P, R

# initial value function
initial_v = np.zeros(101)
initial_v[0] = 0
initial_v[-1]=1

gamma = 1
theta = 1e-6
ph = 0.25

P, R = get_P_and_R_gambler(ph) 

opt_actions,opt_v = valueIteration(P,R,gamma,theta,initial_v,50)

plt.figure(figsize=(8,12))
plt.subplot(211)
plt.plot(opt_actions,'s')
plt.xlabel('capital')
plt.ylabel('stake')
plt.title('p_h = '+str(ph))
plt.grid()
plt.subplot(212)
plt.plot(opt_v,'.-')
plt.xlabel('capital')
plt.ylabel('value')
plt.grid()
plt.savefig('gambler_example_p=' + str(ph) +'.pdf',dpi=180)

