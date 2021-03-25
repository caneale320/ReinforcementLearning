import numpy as np


def policyEval(policy, P, R, gamma, theta, max_iter=1000000):
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

    Note: used inspiration from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
    after getting stuck
    """
    def one_step_lookahead(s, V):
        """
        :param state: current state
        :param v: current value estimator
        :return: A, list of optimal action values under current value estimator
        """
        num_a = policy.shape[1]
        A = np.zeros(num_a)
        for a in range(num_a):
            for s_prime in range(num_S):
                A[a] += P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime])
        return A

    # initialization    
    num_S, num_a = policy.shape
    policy_stable = True

    for s in range(num_S):

        chosen_a = np.argmax(policy[s])

        action_values = one_step_lookahead(s, v)
        best_a = np.argmax(action_values)

        if chosen_a != best_a:
            policy_stable = False

        for i in range(num_a):
            if i != best_a:
                policy[s][i] = 0
            if i == best_a:
                policy[s][best_a] = 1
    return policy, policy_stable



def policyIteration(P,R,gamma,theta,initial_policy,max_iter=1000000):
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
    return policy, v


# I think this works but is really slow due to nested for loops: need to go back and vectorize calculations if time
def valueIteration(P,R,gamma,theta,initial_v,max_iter=1e8):
    """
    This function implements the value iteration algorithm (the in-place version).
    It returns the best action for each state  under a deterministic policy, 
    and the corresponding state-value function.
    """
    print('Running value iteration ...')

    def one_step_lookahead(s, V):
        """
        :param state: current state
        :param v: current value estimator
        :return: A, list of optimal action values under current value estimator
        """
        num_a = num_actions
        num_S = num_states

        A = np.zeros(num_a)

        for a in range(num_a):
            for s_prime in range(num_S):
                A[a] += P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime])
        return A
    
    # initialization
    v = initial_v    
    num_states, num_actions = P.shape[:2]
    k = 0 
    best_actions = [0] * num_states
    delta = 1000

    while delta > theta and k <= max_iter:
        delta = 0
        k += 1
        for s in range(num_states):
            action_values = one_step_lookahead(s, v)
            best_action_value = np.max(action_values)
            delta = max(delta, np.abs(best_action_value - v[s]))
            v[s] = best_action_value
        print(delta)

    for s in range(num_states):
        A = one_step_lookahead(s, v)
        best_actions[s] = np.argmax(A)


    print('number of iterations:', k)
    return best_actions, v

