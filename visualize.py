from algorithms import *
import matplotlib.pyplot as plt

# Look at one episode after training
def see_trained_model(Q_dic, environment, s = (3,0), eps=0, alpha=0.9, gamma=0.9):
    rewards = 0
    steps = [s]
    while s not in [(3,1), (3,2), (3,3), (3, 4)]:
        a = eps_greedy_policy(s, Q_dic, eps)
        # print(a)
        s_2 = tuple(map(sum, zip(s, a)))
        r = calc_reward(environment, s_2)
        possible_actions = determine_possible_actions(s_2)
        Qs2_as = [Q_dic[s_2, a_s] for a_s in possible_actions]
        #print(Qs2_as)
        #Q_dic[(s,a)] = Q_dic[(s,a)] + alpha * (r + gamma * max(Qs2_as) - Q_dic[(s,a)])
        steps.append(s_2)
        s = s_2
        rewards += r
    return steps    


def plot_path(episode, method):
    y = [-i[0]+0.5 for i in episode]
    x = [i[1] +0.5 for i in episode]

    ax = plt.gca()
    plt.plot(x,y)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.xticks([0,1,2,3,4, 5])
    plt.yticks([-4, -3,-2,-1, 0])
    rect = plt.Rectangle((1,-3), width=3, height=1, linewidth=1,edgecolor='r',facecolor='r')

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.title('Episode by ' +  method) 
    plt.show()


def episode_SARSA(Q_dic, environment, s = (3, 0), eps=0, alpha=0.9, gamma=0.9):

    rewards = 0
    steps = [s]
    while s not in [(3,1), (3,2), (3,3), (3, 4)]:
        a = eps_greedy_policy(s, Q_dic, eps)
        s_2 = tuple(map(sum, zip(s, a)))
        r = calc_reward(environment, s_2)
        a_2 = eps_greedy_policy(s_2, Q_dic, eps)
        #Q_dic[(s,a)] = Q_dic[(s,a)] + alpha * (r + gamma * Q_dic[(s_2, a_2)] - Q_dic[(s,a)])
        s = s_2
        a = a_2
        rewards += r
        steps.append(s_2)
    return steps