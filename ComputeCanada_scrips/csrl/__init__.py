from LTL import check_LTL
import time

"""Control Synthesis using Reinforcement Learning.
"""
import numpy as np
from itertools import product
from .mdp import GridMDP
import os
import importlib

if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt
    
if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact



class ControlSynthesis:
    """This class is the implementation of our main control synthesis algorithm.
    
    Attributes
    ----------
    shape : (n_pairs, n_qs, n_rows, n_cols, n_actions)
        The shape of the product MDP.
    
    reward : array, shape=(n_pairs,n_qs,n_rows,n_cols)
        The reward function of the star-MDP. self.reward[state] = 1-discountB if 'state' belongs to B, 0 otherwise.
        
    transition_probs : array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions)
        The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.
    
    Parameters
    ----------
    mdp : mdp.GridMDP
        The MDP that models the environment.
        
    oa : oa.OmegaAutomatan
        The OA obtained from the LTL specification.
        
    discount : float
        The discount factor.
    
    discountB : float
        The discount factor applied to B states.
    
    """
    def __init__(self, mdp, oa, discount=0.99999, discountB=0.99):
        self.mdp = mdp
        self.oa = oa
        self.discount = discount
        self.discountB = discountB  # We can also explicitly define a function of discount
        self.shape = oa.shape + mdp.shape + (len(mdp.A)+oa.shape[1],)
        self.s_vectors = self.state_vectors()
        self.ch_states = self.channeled_states()
        
        # Create the action matrix
        self.A = np.empty(self.shape[:-1],dtype=object)
        for i,q,r,c in self.states():
            self.A[i,q,r,c] = list(range(len(mdp.A))) + [len(mdp.A)+e_a for e_a in oa.eps[q]]
        
        # Create the reward matrix
        self.reward = np.zeros(self.shape[:-1])
        for i,q,r,c in self.states():
            self.reward[i,q,r,c] = 1-self.discountB if oa.acc[q][mdp.label[r,c]][i] else 0
        
        # Create the transition matrix
        self.transition_probs = np.empty(self.shape,dtype=object)  # Enrich the action set with epsilon-actions
        for i,q,r,c in self.states():
            for action in self.A[i,q,r,c]:
                if action < len(self.mdp.A): # MDP actions
                    q_ = oa.delta[q][mdp.label[r,c]]  # OA transition
                    mdp_states, probs = mdp.get_transition_prob((r,c),mdp.A[action])  # MDP transition
                    self.transition_probs[i,q,r,c][action] = [(i,q_,)+s for s in mdp_states], probs  
                else:  # epsilon-actions
                    # print("added epsilon action:", (i,q,r,c),action,"->",(i,action-len(mdp.A),r,c))
                    self.transition_probs[i,q,r,c][action] = ([(i,action-len(mdp.A),r,c)], [1.])
                    
    def states(self):
        """State generator.
        
        Yields
        ------
        state: tuple
            State coordinates (i,q,r,c)).
        """
        n_mdps, n_qs, n_rows, n_cols, n_actions = self.shape
        for i,q,r,c in product(range(n_mdps),range(n_qs),range(n_rows),range(n_cols)):
            yield i,q,r,c
    
    def random_state(self):
        """Generates a random state coordinate.
        
        Returns
        -------
        state: tuple
            A random state coordinate (i,q,r,c).
        """
        n_mdps, n_qs, n_rows, n_cols, n_actions = self.shape
        mdp_state = np.random.randint(n_rows),np.random.randint(n_cols)
        return (np.random.randint(n_mdps),np.random.randint(n_qs)) + mdp_state
    
    def build_gridworld_from_state(self, row, col):
        
        grid_world = np.zeros((5,4))

        grid_world[0, 2] = 2 # b
        grid_world[0, 3] = 3 # d
        grid_world[3, 0] = 4 # a
        grid_world[4, 1] = 5 # c

        grid_world[row, col] = 1 # curr position

        return grid_world
    
    def q_learning(self,start=None,T=None,K=None):
        """Performs the Q-learning algorithm and returns the action values.
        
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
        visited_states = []

        T = T if T else np.prod(self.shape[:-1])
        K = K if K else 100000
        
        Q = np.zeros(self.shape)

        for k in range(K):
            state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
            alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
            epsilon = np.max((1.0*(1 - 1.5*k/K),0.01))
            for t in range(T):
                reward = self.reward[state]
                # if reward>0:
                    # print("!!!!")
                gamma = self.discountB if reward else self.discount
                
                # Follow an epsilon-greedy policy
                if np.random.rand() < epsilon or np.max(Q[state])==0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])
                
                # Observe the next state
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states),p=probs)]
                
                # Q-update
                Q[state][action] += alpha * (reward + gamma*np.max(Q[next_state]) - Q[state][action])
                visited_states.append(state)
                state = next_state
        
        return Q, visited_states
    
    def state_vectors(self):
    
        # assuming state_shape is 4 dim

        size = self.shape[0]*self.shape[1]*self.shape[2]*self.shape[3]

        vec = np.identity(size)
        vectors = {}
        idx = 0
        for i,q,r,c in self.states():
                vectors[(i,q,r,c)] = vec[idx]
                idx += 1
        return vectors
    
    def channeled_states(self):
        # assuming state_shape is 4 dim
        
        size = self.shape[0]*self.shape[1]*self.shape[2]*self.shape[3]

        ch_states = {}
        idx = 0
        for i,q,r,c in self.states():
                ch_s = np.zeros((self.shape[1], self.shape[2], self.shape[3]))
                ch_s[q] = self.mdp.build_grid_world(r, c)
                ch_states[(i,q,r,c)] = np.moveaxis(ch_s, [0,1,2], [2,0,1])
                idx += 1
        return ch_states

    def MC_learning(self, model, LTL_formula, predicates, rewards, C=3, tow=1, n_samples=300,
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

        T = T if T else np.prod(self.shape[:-1])
        K = K if K else 100000
        if search_depth == None: search_depth = T
        success_rate = 0
        # print('visited:', len(visited))
        
        for k in range(K):
            reward = 0
            state_history, channeled_states, action_history, reward_history, better_policy, trajectory = [], [], [], [], [] ,[]
            state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state(empty=True))
            trajectory.append(state[-2]*self.shape[-2]+state[-1])
            state_history.append(state)
            channeled_states.append(self.ch_states[state])
            reward = self.reward[state]
            reward_history.append(reward)
            # if verbose>0: print("N[s_0][:5]:",N[state][:5])
            
            for t in range(T-1):
                
                ###### check if LTL specs are violated
                if len(trajectory)>0 and 'd' in predicates and trajectory[-1] in predicates['d']:break

                MCST_depth = min(T-t-1, search_depth)
                # print(MCST_depth)
                # Choose Action
                t1 = time.time()
                Pi = self.MCTS(model, state, LTL_formula, predicates, trajectory[:-1], state_history, rewards, N, W, Q, P, visited,
                                n_samples=n_samples, depth=MCST_depth, tow=tow, C=C)
                t2 = time.time()
                # print(t2-t1, "MCTS")
                better_policy.append(Pi.copy())
                action = np.random.choice(len(Pi), p=Pi)
                action_history.append(action)
                # get the next state
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states),p=probs)]
                
                if verbose==1:
                    print(action, end=", ")
                elif verbose==2:
                    print(self.build_gridworld_from_state(state[-2], state[-1]), end="\r")
                elif verbose==3:
                    print("step:",k, "MCTS Pi:",Pi)
                    for i in N:
                        print(i, N[i])
                
                state = next_state
                
                reward = self.reward[state]
                trajectory.append(state[-2]*self.shape[-2]+state[-1])
                state_history.append(state)
                channeled_states.append(self.ch_states[state])
                reward_history.append(reward)
                
            outcome = check_LTL(LTL_formula, trajectory, predicates)
            if len(outcome) > 0 and outcome[0]:
                reward = 1
                success_rate += 1
                print("LTL [+++] ", "LDBA [",round(np.sum([rewards[i] for i in state_history]), 2),"]" , "path:", trajectory)
                if verbose>0:
                    print("success ep:",k+1,"/",K)
                    # print("states (if in acc)", [self.oa.acc[q][self.mdp.label[r,c]][0] for (i,q,r,c) in state_history])
                break
            else:
                # print("FAIL: states (if in acc)", [self.oa.acc[q][self.mdp.label[r,c]][0] for (i,q,r,c) in state_history])
                print("LTL [---] ", "LDBA [",round(np.sum([rewards[i] for i in state_history]), 2),"]" , "path:", trajectory)

        if verbose>0:
            print("trajectory:",trajectory)
            print("action_history:",action_history)
            print("state history:", state_history)
            print("----------")
        
        return state_history, channeled_states, trajectory, action_history, reward_history, better_policy

    def MCTS_rec(self, model, root, LTL_formula, trajectory, episode, predicates, rewards, N={}, W={}, Q={}, P={}, visited=set(), C=1, depth=100, random_move_chance=0, foo=0):
        
        LTL_coef = 0.5

        location = root[-2]*self.shape[-2]+root[-1]
        episode.append(root)
        trajectory.append(location)
        # if rewards[root]>0: print("!!!")
        ######## check if LTL specs are violated
        if 'd' in predicates and location in predicates['d']: return -1

        if depth < 1: # search depth limit reached
            ldba_rew = np.sum([rewards[i] for i in episode])
            # ldba_rew = 0
            outcome = check_LTL(LTL_formula, trajectory, predicates)
            if outcome[0]:
                # print("winning traj:", len(trajectory), trajectory)
                return ldba_rew + LTL_coef
            else: return ldba_rew - LTL_coef
        
        elif root not in visited: # unexplored leaf node
            visited.add(root)
            model_input = self.ch_states[root]
            model_output = model(model_input[np.newaxis])
            value = model_output[1].numpy()[0][0]
            P[root] = model_output[0].numpy()[0]
            # ldba_rew = np.sum([rewards[i] for i in episode])
            return value

        ### selecting the next node to expand ###
        U = C * P[root] * (np.sqrt(np.sum(N[root])))/(1+N[root])
        None_idx = self.transition_probs[root]==None
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
            states, probs = self.transition_probs[root][next_move]
        except Exception as e:
            print("exception in observing next state MCTS")
            print(e)
            print("additional info:")
            print("(U + Q[root])",(U + Q[root]))
            print("None_idx:",None_idx)
            print("next move:", next_move)
            print("self.transition_probs[root]", self.transition_probs[root])

        next_state = states[np.random.choice(len(states),p=probs)]

        ### expanding the next move and back tracking ###
        value = self.MCTS_rec(model, next_state, LTL_formula, trajectory, episode, predicates, rewards, N, W, Q, P, visited=visited, C=C, depth=depth-1, foo=foo)
        N[root][next_move] += 1
        W[root][next_move] += value
        Q[root][next_move] = W[root][next_move]/N[root][next_move]

        return value

    def MCTS(self, model, root, LTL_formula, predicates, trajectory, episode, rewards, N, W, Q, P, visited=set(), n_samples=100, tow=1, C=1, depth=100, foo=0):
        
        for sample in range(n_samples):
            self.MCTS_rec(model, root, LTL_formula, trajectory.copy(), episode.copy(), predicates, rewards, N, W, Q, P, visited=visited, C=C, depth=depth, foo=foo)

        Pi = (N[root]**(1/tow)) / np.sum(N[root]**(1/tow))

        if any(np.isnan(Pi)): # for debugging puposes
            print("Warning")
            print("Pi:",Pi)
            print("N[root]:",N[root])
            print("root:", root)
            print("depth:",depth-(len(trajectory)+1))
            print("trajectory", trajectory, "+", root[-2]*self.shape[-2]+root[-1])

        return Pi


    def greedy_policy(self,value):
        """Returns a greedy policy for the given value function.
        
        Parameters
        ----------
        value: array, size=(n_pairs,n_qs,n_rows,n_cols)
            The value function.
        
        Returns
        -------
        policy : array, size=(n_pairs,n_qs,n_rows,n_cols)
            The policy.
        
        """
        policy = np.zeros((value.shape),dtype=np.int)
        for state in self.states():
            action_values = np.empty(len(self.A[state]))
            for i,action in enumerate(self.A[state]):
                action_values[i] = np.sum([value[s]*p for s,p in zip(*self.transition_probs[state][action])])
            policy[state] = self.A[state][np.argmax(action_values)]
        return policy
    
    def value_iteration(self,T=None,threshold=None):
        """Performs the value iteration algorithm and returns the value function. It requires at least one parameter.
        
        Parameters
        ----------
        T : int
            The number of iterations.
        
        threshold: float
            The threshold value to be used in the stopping condition.
        
        Returns
        -------
        value: array, size=(n_mdps,n_qs,n_rows,n_cols)
            The value function.
        """
        value = np.zeros(self.shape[:-1])
        old_value = np.copy(value)
        t = 0  # The time step
        d = np.inf  # The difference between the last two steps
        while (T and t<T) or (threshold and d>threshold):
            value, old_value = old_value, value
            for state in self.states():
                # Bellman operator
                action_values = np.empty(len(self.A[state]))
                for i,action in enumerate(self.A[state]):
                    action_values[i] = np.sum([old_value[s]*p for s,p in zip(*self.transition_probs[state][action])])
                gamma = self.discountB if self.reward[state]>0 else self.discount
                value[state] = self.reward[state] + gamma*np.max(action_values)
            t += 1
            d = np.nanmax(np.abs(old_value-value))
            
        return value
    
    def simulate(self,policy, LTL_formula, predicates, start=None,T=None,plot=True, animation=None):
        """Simulates the environment and returns a trajectory obtained under the given policy.
        
        Parameters
        ----------
        policy : array, size=(n_pairs,n_qs,n_rows,n_cols)
            The policy.
        
        start : int
            The start state of the MDP.
            
        T : int
            The episode length.
        
        plot : bool 
            Plots the simulation if it is True.
            
        Returns
        -------
        episode: list
            A sequence of states
        reward: boolean
            Wheather or not the trajectory satisfies the LTL formula
        """
        T = T if T else np.prod(self.shape[:-1])
        state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
        episode = [state]
        trajectory = [state[-2]*self.shape[-2]+state[-1]]

        for t in range(T):
            states, probs = self.transition_probs[state][policy[state]]
            state = states[np.random.choice(len(states),p=probs)]
            episode.append(state)
            trajectory.append(state[-2]*self.shape[-2]+state[-1])
            
        if plot:
            def plot_agent(t):
                self.mdp.plot(policy=policy[episode[t][:2]],agent=episode[t][2:])
            t=IntSlider(value=0,min=0,max=T-1)
            interact(plot_agent,t=t)
            
        outcome = check_LTL(LTL_formula, trajectory, predicates)

        if animation:
            pad=5
            if not os.path.exists(animation):
                os.makedirs(animation)
            for t in range(T):
                self.mdp.plot(policy=policy[episode[t][:2]],agent=episode[t][2:],save=animation+os.sep+str(t).zfill(pad)+'.png')
                plt.close()
            os.system('ffmpeg -r 3 -i '+animation+os.sep+'%0'+str(pad)+'d.png -vcodec libx264 -y '+animation+'.mp4')
        
        return episode, outcome[0]

    def run_Q_test(self,policy, LTL_formula, predicates, start=None,T=100, runs=100, verbose=1,  animation=None):
        
        print(f"Running {runs} simulations with {T} time-steps...")
        rewards = []
        episodes = []
        for r in range(runs):
            episode, rew =self.simulate(policy, LTL_formula, predicates, start=start,T=T, plot=verbose>=3, animation=animation)
            rewards.append(rew)
            episodes.append(episode)

            if verbose>=1: print("episode",r,"rew:",rew)
            if verbose>=2: print("states (if in acc)", [self.oa.acc[q][self.mdp.label[r,c]][0] for (i,q,r,c) in episode])
        
        print("Test finished with:")
        print('\tsuccess rate:',np.sum(rewards),"/",runs,"=", round((np.sum(rewards)/runs),3))

        return episodes, rewards
        
    def run_MC_test(self,N,Q,P,W, model, LTL_formula, predicates, start=None,T=100, runs=100, verbose=1):
        
        print(f"Running {runs} simulations with {T} time-steps...")
        
        for r in range(runs):
            
            rewards = []
            episodes = []
            state_history, channeled_states, action_history, reward_history, better_policy, trajectory = [], [], [], [], [] ,[]
            trajectory.append(state[-2]*self.shape[-2]+state[-1])
            state_history.append(state)
            channeled_states.append(self.ch_states[state])
            reward = self.reward[state]
            reward_history.append(reward)
            # Choose Action
            Pi = self.MCTS(model, state, LTL_formula, predicates, trajectory[:-1], N, W, Q, P, visited,
                            n_samples=n_samples, depth=search_depth, tow=tow, C=C)
            better_policy.append(Pi.copy())
            action = np.random.choice(len(Pi), p=Pi)
            action_history.append(action)
            
            # get the next state
            states, probs = self.transition_probs[state][action]
            next_state = states[np.random.choice(len(states),p=probs)]
            
            if verbose==1:
                print(action, end=", ")
            elif verbose==2:
                print(self.build_gridworld_from_state(state[-2], state[-1]), end="\r")
            elif verbose==3:
                print("step:",k, "MCTS Pi:",Pi)
                for i in N:
                    print(i, N[i])
            
            state = next_state
            
            reward = self.reward[state]
            trajectory.append(state[-2]*self.shape[-2]+state[-1])
            state_history.append(state)
            channeled_states.append(self.ch_states[state])
            reward_history.append(reward)
            episode, rew =self.simulate(policy, LTL_formula, predicates, start=start,T=T, plot=verbose>=3)
            rewards.append(rew)
            episodes.append(episode)

            if verbose>=1: print("episode",r,"rew:",rew)
            if verbose>=2: print("states (if in acc)", [self.oa.acc[q][self.mdp.label[r,c]][0] for (i,q,r,c) in episode])
        
        print("Test finished with:")
        print('\tsuccess rate:',np.sum(rewards),"/",runs,"=", round((np.sum(rewards)/runs),3))

        return episodes, rewards
    
    def plot(self, value=None, policy=None, iq=None, **kwargs):
        """Plots the values of the states as a color matrix with two sliders.
        
        Parameters
        ----------
        value : array, shape=(n_mdps,n_qs,n_rows,n_cols) 
            The value function.
            
        policy : array, shape=(n_mdps,n_qs,n_rows,n_cols) 
            The policy to be visualized. It is optional.
            
        save : str
            The name of the file the image will be saved to. It is optional
        """
        
        if iq:
            val = value[iq] if value is not None else None
            pol = policy[iq] if policy is not None else None
            self.mdp.plot(val,pol,**kwargs)
        else:
            # A helper function for the sliders
            def plot_value(i,q):
                val = value[i,q] if value is not None else None
                pol = policy[i,q] if policy is not None else None
                self.mdp.plot(val,pol,**kwargs)
            i = IntSlider(value=0,min=0,max=self.shape[0]-1)
            q = IntSlider(value=self.oa.q0,min=0,max=self.shape[1]-1)
            interact(plot_value,i=i,q=q)
