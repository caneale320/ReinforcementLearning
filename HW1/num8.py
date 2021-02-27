import random

# initialize states of the MRP
stateDict = {"Class 1":
                 {"Facebook":[(0, 0.5), -1], "Class 2":[(0.5,1), -2]},
             "Class 2":
                 {"Sleep":[(0, 0.2), 0], "Class 3":[(0.2, 1), -2]},
             "Class 3":
                 {"Pass": [(0, 0.6), 10], "Pub": [(0.4, 1), 1]},
             "Pass":
                 {"Sleep": [(0,1),0]},
             "Pub":
                 {"Class 1": [(0, 0.2), -2], "Class 2": [(0.2, 0.6), -2], "Class 3": [(0.6, 1), -2]},
             "Facebook":
                 {"Facebook": [(0, 0.9), -1], "Class 1": [(0.9, 1), -2]}
             }

# policy will be "always study"
# define states in terms of above policy, formulating the MDP as an MRP

statePolicyDict = {"Class 1":
                 {"Class 2":[(0, 1), -2]},
             "Class 2":
                 {"Class 3":[(0, 1), -2]},
             "Class 3":
                 {"Pass": [(0, 1), 10]},
             "Pass":
                 {"Sleep": [(0,1),0]},
             "Pub":
                 {"Class 1": [(0, 0.2), -2], "Class 2": [(0.2, 0.6), -2], "Class 3": [(0.6, 1), -2]},
             "Facebook":
                 {"Class 1": [(0, 1), -2]}
             }

# intialize episodic reward counter
global totalReward
totalReward = 0


def nextState(currentState):
    """
    determines the next state using a generated random number and the probabilities in the stateDict
    updates the total episode reward by updating global totalReward
    :param currentState: String, the current state of the process, one of the keys of stateDict
    :return:
    """
    global totalReward
    randomNum = random.random()

    for k, v in statePolicyDict[currentState].items():
        if v[0][0] <= randomNum < v[0][1]:
            totalReward += v[1]
            return k

# initialize state tracker list
global statesVisited
statesVisited = ["Class 1"]


def simulate():
    """
    simulates the student MRP using the nextState function
    :return:
    """
    currentState = "Class 1"
    while currentState != "Sleep":
        currentState = nextState(currentState)
        statesVisited.append(currentState)


print(statesVisited, totalReward)


"""
Episode 1:
States: ['Class 1', 'Class 2', 'Class 3', 'Pass', 'Sleep'] 
Actions: ['Study', 'Study', 'Study', 'Study', 'Study']
Total Reward: 6
"""

"""
Episode 2:
States: ['Class 1', 'Class 2', 'Class 3', 'Pass', 'Sleep'] 
Actions: ['Study', 'Study', 'Study', 'Study', 'Study']
Total Reward: 6
"""

"""
Episode 3:
States: ['Class 1', 'Class 2', 'Class 3', 'Pass', 'Sleep']
Actions: ['Study', 'Study', 'Study', 'Study', 'Study'] 
Total Reward: 6
"""