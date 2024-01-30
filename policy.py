"""
Created on Sat May  7 16:51:17 2022

@author: Laurenty Ignacio

# inspired by https://www.geeksforgeeks.org/q-learning-in-python/
"""

from states import *
from IPython.display import clear_output


def coord_state(state):
    i = (state.target[0] + (state.M -1)/2)*state.M + state.target[1] + (state.M-1)/2 
    j = (state.ball[0] + (state.M-1)/2)*state.M + state.ball[1] + (state.M-1)/2
    return int(i), int(j)

def createEpsilonGreedyPolicy(Q, epsilon, num_actions = 8):
    def policyFunction(state):
   
        Action_probabilities = np.ones(num_actions,
                dtype = float) * epsilon / num_actions
        i,j = coord_state(state)
        best_action = np.argmax(Q[i][j])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
   
    return policyFunction


def eval_Q(Q,n=100):
    acc=0
    optim = 0
    for k in range(n):
        g = game()

        while not g.end:
            i,j = coord_state(g.history[-1])
            next_action = np.argmax(Q[i][j])
            g.next(next_action)
        if g.reward[-1]==1:
            acc+=1/n
        optim+= (g.t-g.Tmin)/n
    return acc, optim
        
    

def qLearning(Q_init=None, discount_factor = 1.0,
                            alpha = 0.6, epsilon = 0.1, M=7, limit= 1, symetries=True):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    Q = np.zeros((M**2,M**2,8)) if Q_init is None else Q_init
    optim = 2
    # Keeps track of useful statistics
    stats ={
        "episode_lengths" : [],
        "episode_rewards" : [],
        "episode_acc" : [],
        "episode_optim" :[]
        }
    
    count = 1

    policy = createEpsilonGreedyPolicy(Q, epsilon)
    

    # For every episode/game
    while optim > limit :
        # Reset the environment
        episode = game(M=M)
           
        while not episode.end : 
               
            # get probabilities of all actions from current state
            action_probabilities = policy(episode.history[-1])
   
            # choose action according to 
            # the probability distribution
            action = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
   
            # take action, get reward and transit to next state
            episode.next(action)

            # TD Update
            i,j = coord_state(episode.history[-1])
            best_next_action = np.argmax(Q[i][j])    
            td_target = episode.reward[-1] + discount_factor * Q[i][j][best_next_action]
            i, j = coord_state(episode.history[-2])
            td_delta = td_target - Q[i][j][action]
            Q[i][j][action] += alpha * td_delta
            
            #TD update for symetric games
            if symetries:
                symetrics_episodes = episode.symetries()
                for sym_episode in symetrics_episodes:
                    actions = sym_episode.recover_actions()
                    i, j = coord_state(sym_episode.history[-2])
                    td_delta = td_target - Q[i][j][actions[-1]]
                    Q[i][j][actions[-1]] += alpha * td_delta
        
        # evaluate model
        if count %1000 ==0:
            clear_output(wait=True)
            print('episodes played',count)
            acc, optim = eval_Q(Q,1000)
            stats['episode_acc'].append(acc)
            stats["episode_optim"].append(optim)
        count +=1


        #some stats
        stats["episode_rewards"].append(episode.reward[-1])       
        stats["episode_lengths"].append(episode.t-episode.Tmin)
    print('needed episodes :',count)
        

                
    return Q, stats