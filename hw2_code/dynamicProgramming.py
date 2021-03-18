import numpy as np


def policyEval(policy, P, R, gamma, theta, max_iter=1000):
    """
    This function implements the policy evaluation algorithm (the synchronous
    version) 
    It returns state value function v_pi.
    """
    num_S, num_a = policy.shape
    v = np.zeros(num_S)  # initialize value function
    k = 0  # counter of iteration

    for k in range(int(max_iter)):
        delta = 0
        new_values = np.zeros(num_S)
        # iterate through each state
        for s in range(num_S):
            v_temp = v[s]
            new_Vs_terms = []
            for a in range(len(policy[s])):
                for s_prime in range(len(P[s, a])):
                    new_term = P[s, a, s_prime] * policy[s, a] * (R[s, a, s_prime] + gamma * v[s_prime])
                    new_Vs_terms.append(new_term)

            new_values[s] = sum(new_Vs_terms)
            delta = max(delta, abs(v_temp - new_values[s]))
        v = new_values
        if delta < theta:
            print(k, "iterations")
            return np.around(v, 4)
    return np.around(v, 4)




def policyImprv(P,R,gamma,policy,v):
    """
    This function implements the policy improvement algorithm.
    It returns the improved policy and a boolean variable policy_stable (True
    if the new policy is the same as the old policy)
    """
    # initialization    
    num_S, num_a = policy.shape
    policy_new = np.zeros([num_S,num_a])
    policy_stable = True

    for s in range(num_S):
        if s == 0 or s == 15:
            continue
        potential_returns = []
        for a in range(num_a):
            action_terms = []
            for s_prime in range(num_S):
                new_term = P[s, a, s_prime] * policy[s, a] * (R[s, a, s_prime] + gamma * v[s_prime])
                action_terms.append(new_term)
            potential_returns.append(sum(action_terms))
        potential_returns = np.around(np.array(potential_returns), 5)
        best_actions = [index for index, value in enumerate(potential_returns) if value == max(potential_returns)]

        for i in best_actions:
            policy_new[s, i] = 1/len(best_actions)

        if not(np.array_equal(policy[s], policy_new[s])):
            policy_stable = False

    return policy_new, policy_stable


"""
for some reason i couldn't get this function to converge. It oscillates between the optimal policy in the book and
another policy, without seeming to converge.
"""

def policyIteration(P,R,gamma,theta,initial_policy,max_iter=3):
    """
    This function implements the policy iteration algorithm.
    It returns the final policy and the corresponding state value function v.
    """
    policy_stable = False
    policy = np.copy(initial_policy)
    num_iter = 0
    
    while (not policy_stable) and num_iter < max_iter:
        num_iter += 1
        print('Policy Iteration: ', num_iter)
        # policy evaluation
        v = policyEval(policy,P,R,gamma,theta)
        # policy improvement
        policy, policy_stable = policyImprv(P,R,gamma,policy,v)
        print(policy)
    return policy, v



def valueIteration(P,R,gamma,theta,initial_v,max_iter=1e8):
    """
    This function implements the value iteration algorithm (the in-place version).
    It returns the best action for each state  under a deterministic policy, 
    and the corresponding state-value function.
    """
    print('Running value iteration ...')    
    
    # initialization
    v = initial_v    
    num_states, num_actions = P.shape[:2]
    k = 0 
    best_actions = [0] * num_states

    delta = 0
    for k in range(max_iter):
        for s in range(num_states):
            old_value = v[s]

            potential_returns = []
            for a in range(num_actions):
                action_terms = []
                for s_prime in range(num_states):
                    new_term = P[s, a, s_prime] * (R[s, a, s_prime] + gamma * v[s_prime])
                    action_terms.append(new_term)
                potential_returns.append(sum(action_terms))

            best_actions[s] = potential_returns.index(max(potential_returns))

            v[s] = max(potential_returns)
            delta = max(delta, abs(old_value-v[s]))

        if delta < theta:
            break

    print('number of iterations:', k)
    return best_actions, v

