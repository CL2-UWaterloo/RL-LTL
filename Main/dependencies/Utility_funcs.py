import numpy as np
from dependencies.LTL import *
from dependencies.RL_LTL import RL_LTL
import pickle

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

def build_grid_world(grid_mdp, enc, row, col, agent=True):
    
    grid_world = np.zeros(grid_mdp.shape)
    
    for i in range(grid_mdp.shape[0]):
        for j in range(grid_mdp.shape[1]):
            if grid_mdp.structure[i,j]=='B': grid_world[i,j] = -1 # obsticale
            elif grid_mdp.label[i,j]!=(): grid_world[i,j] = enc.index(grid_mdp.label[i,j])+2 # Empty. look at lables
    if agent:
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

def channeled(csrl, enc, agent=True):
    # assuming state_shape is 4 dim
    
    size = np.prod(csrl.shape)

    ch_states = {}
    idx = 0
    for i,q,r,c in csrl.states():
            ch_s = np.zeros((csrl.shape[1], csrl.shape[2], csrl.shape[3]))
            ch_s[q] = build_grid_world(csrl.mdp, enc, r, c, agent)
            ch_states[(i,q,r,c)] = np.moveaxis(ch_s, [0,1,2], [2,0,1])
            # ch_states[(i,q,r,c)] = ch_s
            idx += 1
    # return np.moveaxis(ch_states, [0,1,2], [2,0,1])
    return ch_states

def update_minecraft_inventory(predicates, trajectory):
    bridge = "<> ((tr /\ <> (ir /\ <> fa)) \/ (ir /\ <> (tr /\ <> fa))) /\ [] (~ wt \/ br)"
    bridge = parser.parse(bridge)
    # axe = ("F (tr & F (wb & F (ir & F ts))) & G (! wt | br)")
    if check_LTL(bridge, trajectory, predicates)[0]:
        predicates['br'] = predicates['wt'] # add bridge on all waters

def MC_learning(csrl, model, LTL_formula, predicates, rewards, ch_states, C=3, tow=1, n_samples=300,
                search_depth=None, N={}, W={}, Q={}, P={}, verbose=0.5, visited=set(),
                start=None,T=None,K=None, run_num=None, ltl_f_rew=None, NN_value_active=False, LTL_coef=10,
                danger_zone='d', reachability=False, best_val_len = {}):
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
    LTL_coef *= 1000
    T = T if T else np.prod(csrl.shape[:-1])
    K = K if K else 100000
    if search_depth == None: search_depth = T
    success_rate = 0
    MCTS_win_rate = []
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
            if len(trajectory)>0 and danger_zone in predicates and trajectory[-1] in predicates[danger_zone]:break

            MCST_depth = min(T-t-1, search_depth)
            # print(MCST_depth)
            # Choose Action
            # print(t+1,")","Depth:",MCST_depth,end=" | ")
            Pi, acc_value, best_val_len[state] = MCTS(csrl, model, state, LTL_formula, predicates.copy(), trajectory[:-1], state_history[:-1], rewards,
                                 ch_states, N, W, Q, P, visited, n_samples=n_samples, depth=MCST_depth, tow=tow, C=C, danger_zone=danger_zone,
                                 ltl_f_rew=ltl_f_rew, NN_value_active=NN_value_active, LTL_coef=LTL_coef, reachability=reachability,
                                 best_val_len = best_val_len)
            if t==0: print(run_num, ") MCTS conf:", round(acc_value, 2), ", det:", round(Pi.max(), 2), end=" | ")
            if csrl.mdp.p > 0.99 and acc_value > 0.99 and Pi.max() > 0.99: n_samples = 1 # a small hack to speedup training in deterministic case
            # NN_value_active = True if acc_value>0 else False
            # print(t2-t1, "MCTS")
            MCTS_win_rate.append(acc_value)
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
                print("step:",t, "MCTS Pi:", [round(i, 2) for i in Pi])
                # for i in N:
                #     print(i)
                #     print(i, N[i])
            
            state = next_state
            trajectory.append(state[-2]*csrl.shape[-2]+state[-1])
            state_history.append(state)
            channeled_states.append(ch_states[state].copy())
            reward_history.append(csrl.reward[state])
            if "wt" in predicates and len(trajectory)>2: update_minecraft_inventory(predicates, trajectory) # only for mine_craft worlds

            if ltl_f_rew:
                reward = check_LTL(LTL_formula, trajectory, predicates)
                reward = reward[0] if len(reward)>0 else False
                if reachability and reward>0: # if the problem is expepicilty a reachability problem, once we reach an acc state we are done.
                    reward_history[-1] += reward # add LTL_f reward to last reward
                    break 
        
        if ltl_f_rew:
            outcome = check_LTL(LTL_formula, trajectory, predicates)
            observed_labels = [csrl.mdp.label[state[-2],state[-1]] for state in state_history]
            if len(outcome) > 0 and outcome[0]:
                reward = 1
                success_rate += 1
                print('s:',trajectory[0], "LTL_f [+++] ", "LDBA [",round(np.sum([rewards[i] for i in state_history]), 2),"]" ,
                      "observed labels:", np.array(observed_labels, dtype=object)[[i != () for i in observed_labels]])
            else:
                print('s:',trajectory[0],"LTL_f [---] ", "LDBA [",round(np.sum([rewards[i] for i in state_history]), 2),"]" ,
                      "observed labels:", np.array(observed_labels, dtype=object)[[i != () for i in observed_labels]] )
        elif reward_history[-1]>0:
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
            pass

    
    if run_num!=None:
        with open(f"outputs/Log_run.txt", 'a') as f:
            f.write(str(run_num))
            f.write(") trajectory: ")
            f.write(', '.join(str(i) for i in trajectory))
            f.write("\n\n action_history: ")
            f.write(', '.join(str(i) for i in action_history))
            f.write("\n\nMCTS win rate: ")
            f.write(', '.join(str(i) for i in [round(i, 2)for i in MCTS_win_rate]))
            f.write("\n\n state history: ")
            f.write(', '.join(str(i) for i in state_history))
            f.write("\n\n reward_history: ")
            f.write(', '.join(str(i) for i in [round(i, 2)for i in reward_history]))
            f.write("\n\n better_policy: ")
            for i in better_policy:
                f.write(", ".join(str(j) for j in [round(j, 2) for j in i]))
                f.write("\n")
            f.write("\n\n----------\n")
    if verbose>0:
        print("reward_history:", [round(i, 2)for i in reward_history])
    if verbose>1:
        print("trajectory:", trajectory)
        print("action_history:",action_history)
        print("state history:", state_history)
        print("----------")
    
    return state_history, channeled_states, trajectory, action_history, reward_history, better_policy, best_val_len

def reshaped_rew(value, len, best_val_len):
    # THIS WAY OF RESHASPING THE REWARDS HAS A PROBLEM. THE FIRST TIME A BETTER VALUE IS SEEN IT RESHAPES
    # INTO A HIGHER REWARD WHICH LATER DAMPENS ALL REHSAPED REWARDS OF THE SAME VALUE. SO REVERTING TO 
    # JUST RETURNING THE VALUE INSTEAD
    # value_star, len_star = best_val_len
    # reshaped_value = (((len_star/len)**2) * (value/value_star)) * value
    # return min(reshaped_value, 1)
    return value

def MCTS_rec(csrl, model, root, LTL_formula, trajectory, episode, predicates, rewards, ch_states, N={}, W={}, Q={}, P={},
             visited=set(), C=1, depth=100, random_move_chance=0, ltl_f_rew=None, NN_value_active=True, LTL_coef=1000, danger_zone='d',
             reachability=False, best_val_len = {}, foo=1):

    location = root[-2]*csrl.shape[-2]+root[-1]
    episode.append(root)
    trajectory.append(location)
    # if rewards[root]>0: print("!!!")
    ######## check if LTL specs are violated
    # print("t",len(episode)+depth)
    if danger_zone in predicates and location in predicates[danger_zone]: return (-0.5, len(trajectory))

    if reachability: # if the problem is expepicilty a reachability problem
        if ltl_f_rew:
            # ldba_rew = np.sum([rewards[i] for i in episode])
            outcome = check_LTL(LTL_formula, trajectory, predicates)
            # ldba_rew = outcome[0]*(1+ldba_rew)
            rew = outcome[0] if len(outcome)>0 else False
            # if root == (0, 1, 3, 0): print(ldba_rew)
        else:
            rew = np.sum([rewards[i] for i in episode])*(LTL_coef*rewards[root])
        if rew>0:
            # print(episode[0], ldba_rew, reshaped, best_val_len)
            # reshaped = reshaped_rew(ldba_rew, len(trajectory), best_val_len)
            # return (reshaped, len(trajectory))
            return (rew, len(trajectory))
            
    if depth < 1: # search depth limit reached
        if ltl_f_rew:
            # ldba_rew = np.sum([rewards[i] for i in episode])
            outcome = check_LTL(LTL_formula, trajectory, predicates)
            # ldba_rew = outcome[0] + ldba_rew
            rew = outcome[0]
        else:
            rew = np.sum([rewards[i] for i in episode])*(1/LTL_coef + LTL_coef*rewards[root])
        # return (reshaped_rew(ldba_rew, len(trajectory), best_val_len), len(trajectory)) if ldba_rew>0 else (-0.5, len(trajectory))
        return (rew, len(trajectory)) if rew>0 else (-0.5, len(trajectory))
        
        # if ldba_rew>0: print("MCTS lookahead:", ldba_rew)


        # 
        # ldba_rew = 0
        # 
    
    # intermdeiate rewards #
    # if foo==0 and rewards[root]>0 and depth+len(episode)<13:
    #     print(trajectory, end="| ")
    #     print(depth, len(episode), end=" ")
    #     return 1

    elif root not in visited: # unexplored leaf node
        visited.add(root)
        model_input = ch_states[root].copy()
        model_output = model(model_input[np.newaxis])
        value = model_output[1].numpy()[0][0]
        P[root] = model_output[0].numpy()[0]
        # ldba_rew = np.sum([rewards[i] for i in episode])
        if NN_value_active: return (value, len(trajectory))
        else: return (0.6, len(trajectory))

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
    if "wt" in predicates and len(trajectory)>2: update_minecraft_inventory(predicates, trajectory) # only for mine_craft worlds

    ### expanding the next move and back tracking ###
    value, len_rollout = MCTS_rec(csrl, model, next_state, LTL_formula, trajectory, episode, predicates, rewards, ch_states, N, W, Q, P,
                     visited=visited, C=C, depth=depth-1, ltl_f_rew= ltl_f_rew, NN_value_active=NN_value_active, LTL_coef=LTL_coef,
                     danger_zone=danger_zone, reachability=reachability, best_val_len = best_val_len, foo=0)
    N[root][next_move] += 1
    W[root][next_move] += value
    Q[root][next_move] = W[root][next_move]/N[root][next_move]

    return (value, len_rollout)

def MCTS(csrl, model, root, LTL_formula, predicates, trajectory, episode, rewards, ch_states, N, W, Q, P, visited=set(),
         n_samples=100, tow=1, C=1, depth=100, ltl_f_rew=None, NN_value_active=True, LTL_coef=1000, danger_zone='d', reachability=False,
         best_val_len = {}, foo=1):
    acc_value = 0
    for sample in range(n_samples):
        value, len_rollout = MCTS_rec(csrl, model, root, LTL_formula, trajectory.copy(), episode.copy(), predicates.copy(), rewards,
                              ch_states, N, W, Q, P, visited=visited, C=C, depth=depth, ltl_f_rew=ltl_f_rew,
                              NN_value_active=NN_value_active, LTL_coef=LTL_coef, danger_zone=danger_zone, reachability=reachability,
                              best_val_len = best_val_len[root], foo=foo)
        if value > best_val_len[root][0] or (value == best_val_len[root][0] and len_rollout < best_val_len[root][1]):
            # print("?", value)
            best_val_len[root] = value, len_rollout
        acc_value += value

    Pi = (N[root]**(1/tow)) / np.sum(N[root]**(1/tow))

    if any(np.isnan(Pi)): # for debugging purposes
        print("Warning")
        print("Pi:",Pi)
        print("N[root]:",N[root])
        print("root:", root)
        print("depth:",depth)
        print("trajectory", trajectory, "+", root[-2]*csrl.shape[-2]+root[-1])

    # print(Pi, acc_value/n_samples, best_val_len)
    return Pi, acc_value/n_samples, best_val_len[root]

def run_Q_test(csrl, policy, LTL_formula, predicates, start=None, T=100, runs=100, verbose=1,  animation=None, reachability=False):
    
    print(f"Running {runs} simulations with {T} time-steps...")
    rewards = []
    episodes = []
    rew_table = (np.ones(csrl.mdp.shape), np.zeros(csrl.mdp.shape))
    start_idx = start if type(start) == np.ndarray else None

    for r in range(runs):
        if not start_idx is None:
            start = start_idx.flatten()[r % len(start_idx)] # this is to make sure all cells are tested at least once
            start = int(start.split(',')[0]), int(start.split(',')[1])
            if rew_table[1][start] > 1: start = None # if already tested, no need to explicitly test again
        
        episode, rew = csrl.simulate(policy, LTL_formula, predicates.copy(), start=start, T=T, plot=verbose>=3, animation=animation)
        
        if reachability:
            trajectory = [s[-2]*csrl.shape[-2]+s[-1] for s in episode]
            # rew = check_LTL(LTL_formula, trajectory, predicates)[0]
            # rew = any([i>0 for i in [csrl.reward[x] for x in episode]])
            rewards.append(rew)
        else:
            rewards.append([csrl.reward[x] for x in episode])

        episodes.append(episode)

        rew_table[0][episode[0][-2], episode[0][-1]] += 1
        rew_table[1][episode[0][-2], episode[0][-1]] += rew
        if rew_table[1][episode[0][-2], episode[0][-1]] == 1: rew_table[1][episode[0][-2], episode[0][-1]] += 1 # to correct 0.99 issue

        if verbose>=1: print("episode",r,"rew:",rew)
        if verbose>=2: print("episode:",episode)
        if verbose>=3: print("states (if in acc)", [csrl.oa.acc[q][csrl.mdp.label[r,c]][0] for (i,q,r,c) in episode])
    
    print("Test finished with:")
    print('\tsuccess rate:',np.sum(rewards),"/",runs,"=", round((np.sum(rewards)/runs),3))

    return episodes, rewards, rew_table[1]/rew_table[0]

def decayed_reward(size, init_rew, l=0.95):
  
  rew_seq = []

  for i in range(size):
    rew_seq.append(init_rew)
    init_rew *= l
  
  return np.array(rew_seq)
  
def eval(model, n_gw_test=10, n_runs = 1, training_epochs = 10):
    baseline = np.array([
       [0.0652, 0.0656, 0.2148, 0.2816, 0.3776, 0.4112, 0.5716, 0.5844, 0.5948, 0.7012, 0.701 ],
       [0.3184, 0.434 , 0.5752, 0.5924, 0.6708, 0.6912, 0.6836, 0.6872, 0.796 , 0.8084, 0.8046],
       [0.1096, 0.2284, 0.244 , 0.434 , 0.4936, 0.494 , 0.5456, 0.5936, 0.6012, 0.726 , 0.7372],
       [0.    , 0.0832, 0.2672, 0.3844, 0.454 , 0.5052, 0.5472, 0.564 , 0.588 , 0.624 , 0.6204],
       [0.0996, 0.4336, 0.5184, 0.528 , 0.5604, 0.5952, 0.5936, 0.7132, 0.7216, 0.7436, 0.738 ],
       [0.208 , 0.4076, 0.4384, 0.526 , 0.6128, 0.6432, 0.6588, 0.72  , 0.7296, 0.7844, 0.8122],
       [0.    , 0.09  , 0.206 , 0.2208, 0.2424, 0.388 , 0.3992, 0.5132, 0.6416, 0.6496, 0.6506],
       [0.0348, 0.0608, 0.1748, 0.3244, 0.3292, 0.45  , 0.4996, 0.5276, 0.5596, 0.6232, 0.6472],
       [0.0168, 0.0572, 0.0656, 0.1316, 0.2108, 0.228 , 0.2796, 0.3528, 0.3712, 0.5004, 0.5464],
       [0.1096, 0.3336, 0.4776, 0.6212, 0.7104, 0.7704, 0.8092, 0.8176, 0.816 , 0.8392, 0.8366]])
    
    accuracies = []
    for i in range(n_gw_test):
        temp = []
        with open(f'outputs/gws/test/gw{i+1}.dat', 'rb') as f:
            gw = pickle.load(f)
        for j in range(n_runs):
            env = RL_LTL(gw, model, kwargs={'training': False})
            env.train(training_epochs, smart_start=True)
            env.get_policy(1, reset_tables=False)
            temp.append([0] * (training_epochs - len(env.policy_succ_rate) + 1) + env.policy_succ_rate)

        accuracies.append(np.mean(temp, 0))

    accuracies = np.array(accuracies)
    
    speedup = (accuracies - baseline[:n_gw_test]).mean()
    perf = (accuracies[:, -1] - baseline[:n_gw_test, -1]).mean()

    return speedup, perf