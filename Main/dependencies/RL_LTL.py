import time
import os
import numpy as np
from dependencies.Utility_funcs import MC_learning, run_Q_test
import pickle

class RL_LTL:
    def __init__(self, gw, model, **kwargs):
        self.gw = gw
        self.model = model
        self.visited_states_train = []
        self.visited_states_test = []
        self.LTL_coef = 1
        self.NN_value_active = False
        self.start_idx = None
        self.policy_succ_rate = []
        self.N, self.W, self.Q, self.P, self.visited_test, self.visited_train = np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), set(), set()
        self.gw_idx = np.array([[f'{j},{i}' for i in range(gw.mdp.shape[1])] for j in range(gw.mdp.shape[0])])
        self.empty_idx = (self.gw.structure == 'E')
        self.unlabeled_idx = np.array([[i==() for i in x] for x in self.gw.label])

        self.search_depth = kwargs['search_depth'] if 'search_depth' in kwargs else 200 
        self.MCTS_samples = kwargs['MCTS_samples'] if 'MCTS_samples' in kwargs else 100

        self.num_training_epochs = kwargs['num_training_epochs'] if 'num_training_epochs' in kwargs else 10
        self.num_test_epochs = kwargs['num_test_epochs'] if 'num_test_epochs' in kwargs else 20
        self.training = kwargs['training'] if 'training' in kwargs else True
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 10
        self.C = kwargs['C'] if 'C' in kwargs else 0.5
        self.tow = kwargs['tow'] if 'tow' in kwargs else 0.2
        self.K = kwargs['K'] if 'K' in kwargs else 1
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32
        self.steps_per_epoch = kwargs['steps_per_epoch'] if 'steps_per_epoch' in kwargs else 4
        self.danger_zone = kwargs['danger_zone'] if 'danger_zone' in kwargs else 'd'
        self.best_val_len = {}
        for s in gw.csrl.states(): self.best_val_len[s] = (0.001, 99999)
        if os.path.exists("outputs/Log_run.txt"): os.remove("outputs/Log_run.txt")

    def train(self, num_training_epochs=None, start=None, T =[50], smart_start=False, **kwargs):
        if num_training_epochs != None: self.num_training_epochs = num_training_epochs
        idx = 0
        C = self.C
        self.train_history = []
        start_idx = None
        stop_acc = kwargs['stop_acc'] if 'stop_acc' in kwargs else 1

        for i in T:
            idx += 1
            print("##########################")
            print("C:",C, "| tow:",self.tow)
            # TRAIN ##############################
            train_wins = 0
            # num_training_epochs = int(200 - 1.9*i)
            # model = build_model(ch_states[(0,0,0,0)].shape, csrl.shape[-1])
            N, W, Q, P, visited_train = np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), set()
            for epoch in range(self.num_training_epochs):
                t1 = time.time()
                state_history, channeled_states, trajectory, action_history, reward_history, better_policy, best_val_len = MC_learning(
                    self.gw.csrl, self.model, self.gw.LTL_formula, self.gw.predicates.copy(), self.gw.csrl.reward, self.gw.ch_states,
                    N = N, W = W, Q = Q, P = P, C=C, tow=self.tow, n_samples=self.MCTS_samples, visited=visited_train,
                    start=start, search_depth=self.search_depth, verbose=0, T=i, K=self.K, NN_value_active=self.NN_value_active,
                    run_num=epoch, ltl_f_rew=True, reachability=True, best_val_len = self.best_val_len, danger_zone=self.danger_zone)
                
                self.visited_states_train += state_history
                t2 = time.time()
                # print(t2-t1, " run episode")

                # win = check_LTL(self.gw.LTL_formula, trajectory, self.gw.predicates)[0]
                win = reward_history[-1]
                if win:
                    train_wins+=1
                    self.NN_value_active = True

                if self.training and len(action_history)>0:
                    if epoch==0:
                        x_train = np.array(channeled_states)[:-1]
                        y1_train = np.array(better_policy)
                        y2_train = np.array(reward_history) + self.LTL_coef*reward_history[-1]
                        # y2_train = np.array(reward_history)
                        y2_train = y2_train[:-1]
                    else:
                        x_train = np.concatenate((x_train, np.array(channeled_states)[:-1]),0)
                        y1_train = np.concatenate((y1_train, np.array(better_policy)),0)
                        y2_train_curr = np.array(reward_history) + self.LTL_coef*reward_history[-1]
                        # y2_train_curr = np.array(reward_history)
                        y2_train = np.concatenate((y2_train, y2_train_curr[:-1]),0)
                    t3= time.time()
                    # print(t3-t2, " build database")
                    tr_hist = self.model.fit(x_train, [y1_train, y2_train], epochs=self.epochs, batch_size=self.batch_size,
                                        steps_per_epoch=self.steps_per_epoch if len(x_train)>self.steps_per_epoch*self.epochs*self.batch_size else None, verbose=0)
                    self.train_history += tr_hist.history['loss']
                self.Q, self.N, self.W, self.P = Q, N, W, P

                if smart_start and self.NN_value_active:
                    self.evaluate(len=i, runs=500)
                    unsolved_idx = (self.rew_table != 1)
                    start_idx = unsolved_idx * self.empty_idx * self.unlabeled_idx
                    if start_idx.sum() > 0 and self.policy_succ_rate[-1]<stop_acc: # at least one starting point
                        start = np.random.choice(self.gw_idx[start_idx])
                        start = int(start.split(',')[0]), int(start.split(',')[1])
                        C *= 1.02
                        # print(len(self.gw_idx[start_idx]), start)
                    else: break # training complete

                # win_hist.append(win)
                t4 = time.time()
                # print(t4-t3, "fit", len(x_train))
            print("Train wins:",train_wins,"/", epoch)


    def get_policy(self, num_test_epochs=None, start=None, reset_tables=True, T =[50], smart_start=False):
        if num_test_epochs != None: self.num_test_epochs = num_test_epochs
        for i in T:
            self.success_rates = []
            self.succes_std = []
            self.win_hist = []

            # TEST ##############################
            test_wins = 0
            if reset_tables:
                self.N, self.W, self.Q, self.P, visited_test = np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), set()
            N, W, Q, P, visited_test = self.N, self.W, self.Q, self.P, self.visited_test
            for epoch in range(self.num_test_epochs):
                state_history, channeled_states, trajectory, action_history, reward_history, better_policy, best_val_len = MC_learning(
                    self.gw.csrl, self.model, self.gw.LTL_formula, self.gw.predicates.copy(), self.gw.csrl.reward, self.gw.ch_states,
                    N = N, W = W, Q = Q, P = P, C=0.1, tow=0.05, n_samples=self.MCTS_samples, visited=visited_test, start=start,
                    search_depth=self.search_depth, verbose=0, T=i, K=1, NN_value_active=True, run_num=epoch, ltl_f_rew=True,
                    reachability=True, best_val_len = self.best_val_len, danger_zone=self.danger_zone)

                # win = check_LTL(LTL_formula, trajectory, predicates)[0]
                win = reward_history[-1]
                if win: test_wins+=1
                self.win_hist.append(win)
                self.visited_states_test += state_history
                self.Q, self.N, self.W, self.P, self.visited_test = Q, N, W, P, visited_test

                if smart_start:
                    self.evaluate(len=i)
                    unsolved_idx = (self.rew_table != 1)
                    start_idx = unsolved_idx * self.empty_idx * self.unlabeled_idx
                    if start_idx.sum() > 0: # at least one starting point
                        start = np.random.choice(self.gw_idx[start_idx])
                        start = int(start.split(',')[0]), int(start.split(',')[1])
                        
            self.success_rates.append(100*test_wins/self.num_test_epochs)
            temp = np.zeros(self.num_test_epochs)
            temp[:test_wins]=1
            std = np.sqrt(self.num_test_epochs*np.var(temp))
            self.succes_std.append((self.success_rates[-1]-std, self.success_rates[-1]+std))
            
            ###############################################################
            print("Test wins:",test_wins,"/",self.num_test_epochs)
            # print("last reward:", reward_history[-1], "  | trajectory:", trajectory)
            # print("Actions:", action_history)

        self.encode_visited_states_test = [i[1]*self.gw.csrl.shape[-2]*self.gw.csrl.shape[-3]+i[2]*self.gw.csrl.shape[-2]+i[3] for i in self.visited_states_test]
        self.encode_visited_states_train = [i[1]*self.gw.csrl.shape[-2]*self.gw.csrl.shape[-3]+i[2]*self.gw.csrl.shape[-2]+i[3] for i in self.visited_states_train]
        self.Q, self.N, self.W, self.P = Q, N, W, P
        self.policy = np.argmax(self.N,axis=4)
        self.value=np.max(self.Q,axis=4)
        self.evaluate(len=i)
        # u, d, r, l

    def evaluate(self, start=None, len=50, runs=1000, verbose=0, animation=None):
        self.policy = np.argmax(self.N,axis=4)
        # self.value=np.max(self.Q,axis=4)
        episodes, rew, rew_table = run_Q_test(self.gw.csrl, self.policy, self.gw.LTL_formula, self.gw.predicates.copy(),
                                   start=start, T=len, runs=runs, verbose=verbose, reachability=True, animation=animation)
        self.policy_succ_rate.append(np.sum(rew)/runs)
        self.rew_table = rew_table
    
      
def eval(model, n_gw_test=10, n_runs = 1, training_epochs = 10):
    """
    inp
    """
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
