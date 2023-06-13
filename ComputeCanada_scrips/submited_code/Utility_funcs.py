import numpy as np
from LTL import *

def get_predicates(grid_mdp):
        aps = list(np.unique(grid_mdp.label))
        aps.pop(aps.index(()))
        aps = [str(x[0]) for x in aps]
        predicates = {}
        for i in aps:
            predicates[i] = []

        index = 0
        for i in grid_mdp.label:
            for j in i:
                if len(j)>0 and str(j[0]) in aps:
                    predicates[str(j[0])].append(index)
                index += 1
        
        return predicates

def build_grid_world(grid_mdp, enc, row, col):
    
    grid_world = np.zeros(grid_mdp.shape)
    
    for i in range(grid_mdp.shape[0]):
        for j in range(grid_mdp.shape[1]):
            if grid_mdp.structure[i,j]=='B': grid_world[i,j] = -1 # obsticale
            elif grid_mdp.label[i,j]!=(): grid_world[i,j] = enc.index(grid_mdp.label[i,j])+2 # Empty. look at lables
    
    grid_world[row, col] = 1
    
    return grid_world

def build_gridworld_from_state(row, col):
        
    grid_world = np.zeros((5,4))

    grid_world[0, 2] = 2 # b
    grid_world[0, 3] = 3 # d
    grid_world[3, 0] = 4 # a
    grid_world[4, 1] = 5 # c

    grid_world[row, col] = 1 # curr position

    return grid_world

def state_vectors(csrl):
    
    # assuming state_shape is 4 dim

    size = csrl.shape[0]*csrl.shape[1]*csrl.shape[2]*csrl.shape[3]

    vec = np.identity(size)
    vectors = {}
    idx = 0
    for i,q,r,c in csrl.states():
            vectors[(i,q,r,c)] = vec[idx]
            idx += 1
    return vectors

def channeled_states(csrl, enc):
    # assuming state_shape is 4 dim
    
    size = csrl.shape[0]*csrl.shape[1]*csrl.shape[2]*csrl.shape[3]

    ch_states = {}
    idx = 0
    for i,q,r,c in csrl.states():
            ch_s = np.zeros((csrl.shape[1], csrl.shape[2], csrl.shape[3]))
            ch_s[q] = build_grid_world(csrl.mdp, enc, r, c)
            ch_states[(i,q,r,c)] = np.moveaxis(ch_s, [0,1,2], [2,0,1])
            idx += 1
    return ch_states

def MC_learning(csrl, model, LTL_formula, predicates, rewards, ch_states, C=3, tow=1, n_samples=300,
                search_depth=None, N={}, W={}, Q={}, P={}, verbose=0.5, visited=set(),
                start=None,T=None,K=None):
    """Performs the MC-learning algorithm and returns the action values.
    
    Parameters
    ----------
    start : int
        The start state of the MDP.
        
    T : int
        The episode length.
    
    K : int 
        The number of episodes.
        
    Returns
    -------
    Q: array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions) 
        The action values learned.
    """

    T = T if T else np.prod(csrl.shape[:-1])
    K = K if K else 100000
    if search_depth == None: search_depth = T
    success_rate = 0
    # print('visited:', len(visited))
    
    for k in range(K):
        reward = 0
        state_history, channeled_states, action_history, reward_history, better_policy, trajectory = [], [], [], [], [] ,[]
        if start: state = (csrl.shape[0]-1,csrl.oa.q0) + start
        else:
            state = csrl.mdp.random_state()
            while(csrl.mdp.structure[state]!='E' or csrl.mdp.label[state]!=()):
                state = csrl.mdp.random_state()
            state = (csrl.shape[0]-1,csrl.oa.q0) + state

        trajectory.append(state[-2]*csrl.shape[-2]+state[-1])
        state_history.append(state)
        channeled_states.append(ch_states[state].copy())
        reward = csrl.reward[state]
        reward_history.append(reward)
        # if verbose>0: print("N[s_0][:5]:",N[state][:5])
        
        for t in range(T-1):
            
            ###### check if LTL specs are violated
            if len(trajectory)>0 and 'd' in predicates and trajectory[-1] in predicates['d']:break

            MCST_depth = min(T-t-1, search_depth)
            # print(MCST_depth)
            # Choose Action
            Pi = MCTS(csrl, model, state, LTL_formula, predicates, trajectory[:-1], state_history, rewards, ch_states, N, W, Q, P, visited,
                            n_samples=n_samples, depth=MCST_depth, tow=tow, C=C)
            # print(t2-t1, "MCTS")
            better_policy.append(Pi.copy())
            action = np.random.choice(len(Pi), p=Pi)
            action_history.append(action)
            # get the next state
            states, probs = csrl.transition_probs[state][action]
            next_state = states[np.random.choice(len(states),p=probs)]
            
            if verbose==1:
                print(action, end=", ")
            elif verbose==2:
                print(build_gridworld_from_state(state[-2], state[-1]), end="\r")
            elif verbose==3:
                print("step:",k, "MCTS Pi:",Pi)
                for i in N:
                    print(i, N[i])
            
            state = next_state
            
            reward = csrl.reward[state]
            trajectory.append(state[-2]*csrl.shape[-2]+state[-1])
            state_history.append(state)
            channeled_states.append(ch_states[state].copy())
            reward_history.append(reward)
            
        outcome = check_LTL(LTL_formula, trajectory, predicates)
        if len(outcome) > 0 and outcome[0]:
            reward = 1
            success_rate += 1
            print("LTL [+++] ", "LDBA [",round(np.sum([rewards[i] for i in state_history]), 2),"]" , "path:", trajectory)
            if verbose>0:
                print("success ep:",k+1,"/",K)
                # print("states (if in acc)", [csrl.oa.acc[q][csrl.mdp.label[r,c]][0] for (i,q,r,c) in state_history])
            break
        else:
            # print("FAIL: states (if in acc)", [csrl.oa.acc[q][csrl.mdp.label[r,c]][0] for (i,q,r,c) in state_history])
            print("LTL [---] ", "LDBA [",round(np.sum([rewards[i] for i in state_history]), 2),"]" , "path:", trajectory)

    if verbose>0:
        print("trajectory:",trajectory)
        print("action_history:",action_history)
        print("state history:", state_history)
        print("----------")
    
    return state_history, channeled_states, trajectory, action_history, reward_history, better_policy

def MCTS_rec(csrl, model, root, LTL_formula, trajectory, episode, predicates, rewards, ch_states, N={}, W={}, Q={}, P={}, visited=set(), C=1, depth=100, random_move_chance=0, foo=0):
    
    LTL_coef = 1

    location = root[-2]*csrl.shape[-2]+root[-1]
    episode.append(root)
    trajectory.append(location)
    # if rewards[root]>0: print("!!!")
    ######## check if LTL specs are violated
    if 'd' in predicates and location in predicates['d']: return -1

    if depth < 1: # search depth limit reached
        # ldba_rew = np.sum([rewards[i] for i in episode])
        # ldba_rew = 0
        outcome = check_LTL(LTL_formula, trajectory, predicates)
        if outcome[0]:
            # print("winning traj:", len(trajectory), trajectory)
            return LTL_coef
        else: return 0
    
    elif root not in visited: # unexplored leaf node
        visited.add(root)
        model_input = ch_states[root].copy()
        model_output = model(model_input[np.newaxis])
        value = model_output[1].numpy()[0][0]
        P[root] = model_output[0].numpy()[0]
        # ldba_rew = np.sum([rewards[i] for i in episode])
        return value

    ### selecting the next node to expand ###
    U = C * P[root] * (np.sqrt(np.sum(N[root])))/(1+N[root])
    None_idx = csrl.transition_probs[root]==None
    try:
        # print(root)
        temp = (U + Q[root])
        temp[None_idx] = -99999
        next_move = temp.argmax()
        # print(temp[:4])
    except Exception as e:
        print("exception in finding next move MCTS")
        print(e)
        print("additional info:")
        print("U:",U)
        print("root:",root)
        print("W:", W)
        print("None_id:",None_idx)

    ### creating the next subtree ###
    try:
        states, probs = csrl.transition_probs[root][next_move]
    except Exception as e:
        print("exception in observing next state MCTS")
        print(e)
        print("additional info:")
        print("(U + Q[root])",(U + Q[root]))
        print("None_idx:",None_idx)
        print("next move:", next_move)
        print("csrl.transition_probs[root]", csrl.transition_probs[root])

    next_state = states[np.random.choice(len(states),p=probs)]

    ### expanding the next move and back tracking ###
    value = MCTS_rec(csrl, model, next_state, LTL_formula, trajectory, episode, predicates, rewards, ch_states, N, W, Q, P, visited=visited, C=C, depth=depth-1, foo=foo)
    N[root][next_move] += 1
    W[root][next_move] += value
    Q[root][next_move] = W[root][next_move]/N[root][next_move]

    return value

def MCTS(csrl, model, root, LTL_formula, predicates, trajectory, episode, rewards, ch_states, N, W, Q, P, visited=set(), n_samples=100, tow=1, C=1, depth=100, foo=0):
    
    for sample in range(n_samples):
        MCTS_rec(csrl, model, root, LTL_formula, trajectory.copy(), episode.copy(), predicates, rewards, ch_states, N, W, Q, P, visited=visited, C=C, depth=depth, foo=foo)

    Pi = (N[root]**(1/tow)) / np.sum(N[root]**(1/tow))

    if any(np.isnan(Pi)): # for debugging puposes
        print("Warning")
        print("Pi:",Pi)
        print("N[root]:",N[root])
        print("root:", root)
        print("depth:",depth-(len(trajectory)+1))
        print("trajectory", trajectory, "+", root[-2]*csrl.shape[-2]+root[-1])

    return Pi

def run_Q_test(csrl, policy, LTL_formula, predicates, start=None,T=100, runs=100, verbose=1,  animation=None):
        
    print(f"Running {runs} simulations with {T} time-steps...")
    rewards = []
    episodes = []
    for r in range(runs):
        episode, rew =csrl.simulate(policy, LTL_formula, predicates, start=start,T=T, plot=verbose>=3, animation=animation)
        rewards.append(rew)
        episodes.append(episode)

        if verbose>=1: print("episode",r,"rew:",rew)
        if verbose>=2: print("states (if in acc)", [csrl.oa.acc[q][csrl.mdp.label[r,c]][0] for (i,q,r,c) in episode])
    
    print("Test finished with:")
    print('\tsuccess rate:',np.sum(rewards),"/",runs,"=", round((np.sum(rewards)/runs),3))

    return episodes, rewards