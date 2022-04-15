import numpy as np
from collections import defaultdict
from tqdm import tqdm

def def_value():
    return 0


def def_list():
    return []


def determine_possible_actions(state):
    actions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    if state[0] == 0:
        actions.remove((-1, 0))
    if state[1] == 0:
        actions.remove((0, -1))
    if state[1] == 4:
        actions.remove((0, 1))
    if state[0] == 3:
        actions.remove((1, 0))
    return actions



def eps_greedy_policy(state, Q, eps):
    p = np.random.random()
    actions = determine_possible_actions(state)
    if p < eps:
        idx = np.random.choice(len(actions), 1)
        a = actions[idx[0]]
    else:
        actions_Q = defaultdict(def_list)  
        for possible_action in actions:
            actions_Q[Q[(state, possible_action)]].append(possible_action)
        best_Q = max(actions_Q)
        best_actions = actions_Q[best_Q]
        # print(best_Q, best_actions)
        idx = np.random.choice(len(best_actions), 1)
        a = best_actions[idx[0]]
    return a


def calc_reward(environment, place):
    return environment[place]


def Q_Learning(environment, terminal_states, episodes, alpha, gamma, eps):
    Q_dic = defaultdict(def_value)
    cum_rewards = []
    for ep in tqdm(range(episodes)):
        s = (3, 0)
        rewards = 0
        while s not in terminal_states:
            a = eps_greedy_policy(s, Q_dic, eps)
            s_2 = tuple(map(sum, zip(s, a)))
            r = calc_reward(environment, s_2)
            possible_actions = determine_possible_actions(s_2)
            Qs2_as = [Q_dic[s_2, a_s] for a_s in possible_actions]
            Q_dic[(s,a)] = Q_dic[(s,a)] + alpha * (r + gamma * max(Qs2_as) - Q_dic[(s,a)])
            s = s_2
            rewards += r
        cum_rewards.append(rewards)
    return cum_rewards, Q_dic


def SARSA_Learning(environment, terminal_states, episodes, alpha, gamma, eps):
    Q_dic = defaultdict(def_value)
    cum_rewards = []
    for ep in tqdm(range(episodes)):
        s = (3, 0)
        rewards = 0
        while s not in terminal_states:
            a = eps_greedy_policy(s, Q_dic, eps)
            s_2 = tuple(map(sum, zip(s, a)))
            r = calc_reward(environment, s_2)
            a_2 = eps_greedy_policy(s_2, Q_dic, eps)
            Q_dic[(s,a)] = Q_dic[(s,a)] + alpha * (r + gamma * Q_dic[(s_2, a_2)] - Q_dic[(s,a)])
            s = s_2
            a = a_2
            rewards += r
        cum_rewards.append(rewards)
    return cum_rewards, Q_dic


def nStep_SARSA_Learning(environment, episodes, alpha, gamma, eps, n):
    Q_dic = defaultdict(def_value)
    cum_rewards = []
    for ep in tqdm(range(episodes)):
        #print("Episode", ep)
        T = np.inf
        t = 0
        tao = t - n + 1
        s = (3, 0)
        rewards = 0
        bufferS = []
        bufferA = []
        bufferR = []
        a = eps_greedy_policy(s, Q_dic, eps)
        while tao != T - 1:

            a_2 = None
            if t < T:
                s_2 = tuple(map(sum, zip(s, a)))
                r = calc_reward(environment, s_2)
                bufferR.append(r)
                bufferA.append(a)
                bufferS.append(s)
                if s_2 in [(3,1), (3,2), (3,3), (3, 4)]:
                    T = t + 1
                else:
                    a_2 = eps_greedy_policy(s_2, Q_dic, eps)
            tao = t - n + 1
            if tao >= 0:
                G = 0 
                for i in range(tao, min(tao + n, T)):
                    G += gamma **(i - tao - 1) * bufferR[i]
                if tao + n < T:
                    G = G + gamma**n * Q_dic[(bufferS[tao + n - 1], bufferA[tao + n - 1])]
                Q_dic[((bufferS[tao - 1], bufferA[tao -1]))] = Q_dic[(bufferS[tao -1], bufferA[tao -1])] + alpha * (G - Q_dic[(bufferS[tao -1], bufferA[tao-1])])
            t = t + 1
            s = s_2
            if a_2 is not None:
                a = a_2
    return Q_dic