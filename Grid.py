import numpy as np
from collections import defaultdict


def def_value():
    return 0


def determine_possible_actions(state):
    actions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    if state[0] == 0:
        actions.remove((-1, 0))
    if state[1] == 0:
        actions.remove((0, -1))
    if state[1] == 3:
        actions.remove((0, 1))
    if state[0] == 3:
        actions.remove((1, 0))
    return actions


def policy(state, Q, eps):
    p = np.random.random()
    actions = determine_possible_actions(state)
    print("Possible actions are: ", actions)
    a = (0, 0)
    print("P is ", round(p, 2))
    if p < eps:
        a = np.random.choice(actions, 1)
    else:
        max_Q = -np.inf
        best_actions = []
        for possible_action in actions:
            s_2 = tuple(map(sum, zip(state, possible_action)))

            if Q[(s_2, possible_action)] >= max_Q:
                print("New possible action")
                best_actions.append(possible_action)
        idx = np.random.choice(len(best_actions), 1)
        a = best_actions[idx]
    print("Action to take is", a)
    return a


def calc_reward(environment, place):
    return environment[place]


def Q_Learning(environment, episodes, alpha, gamma, eps):
    Q_dic = defaultdict(def_value)
    for ep in range(episodes):
        s = (3, 0)
        while s != (3, 4):
            a = policy(s, Q_dic, eps)
            s_2 = s + a
            r = calc_reward(environment, s_2)
            Q_dic[(s,a)] = Q_dic[(s,a)] + alpha * (r + gamma * np.argmax(Q_dic) - Q_dic[(s,a)])
            s = s_2


environment = np.array([[-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -100, -100, -100, 0]
                        ])
Q_dic = defaultdict(def_value)
policy((0,0), Q_dic, 0.2)