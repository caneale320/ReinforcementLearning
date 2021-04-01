import math

import numpy as np

def TD0(get_episode,policy, initial_v, gamma, alpha,num_episodes = 1):
# This function implements TD(0).
# get_episode: function to generate an episode
# policy: the policy to be evaluated 
# initial_v: initial estimate for value function v
# gamma: discount factor
# alpha: learning rate
# num_episodes: number of episodes (iterations)
# The function returns the estimate of v

    # initialization  
    v = np.copy(initial_v)
    
    for episode in range(num_episodes):
        states, actions, rewards = get_episode(policy)
        for i in range(len(states) - 1):
            v[states[i]] = v[states[i]] + alpha*(rewards[i] + gamma*v[states[i+1]] - v[states[i]])

    return v


def TD_n(get_episode, policy, initial_v, n, gamma, alpha,num_episodes = 1):
# This function implements n-step TD.
# get_episode: function to generate an episode
# policy: the policy to be evaluated 
# initial_v: initial estimate for value function v
# n: number of steps to look ahead
# gamma: discount factor
# alpha: learning rate
# num_episodes: number of episodes (iterations)
# The function returns the estimate of v
    def n_step_return(tau, n, T, rewards):
        G = 0
        for i in range(tau + 1, min((tau + n), T)):
            G = G + gamma**(i-tau-1) * rewards[i]
        return G

    # initialization
    v = np.copy(initial_v)
    
    for episode in range(num_episodes):
        t = 0
        states, actions, rewards = get_episode(policy)
        T = len(states) - 1

        for t in range(T):
            tau = t - n + 1
            if tau >= 0:
                G = n_step_return(tau, n, T, rewards)
                if tau + n < T:
                    G = G + gamma**n*v[states[tau + n]]
                v[states[tau]] = v[states[tau]] + alpha*(G - v[states[tau]])
            if tau == T - 1:
                break

    return v


def TD_lambda(get_episode, policy, initial_v, lambda_, gamma, alpha,
              num_episodes=1):
# This function implements TD_lambda (backward view).
# get_episode: function to generate an episode
# policy: the policy to be evaluated 
# initial_v: initial estimate for value function v
# lambda_: value of lambda in TD(lambda)
# gamma: discount factor
# alpha: learning rate
# num_episodes: number of episodes (iterations)
# The function returns the estimate of v
              
    # initialization 
    v = np.copy(initial_v)
    
    """
    Your Code
    """

    return v


        