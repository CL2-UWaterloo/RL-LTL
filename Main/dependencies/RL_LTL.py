import time
import os
import numpy as np
from dependencies.Utility_funcs import MC_learning

class RL_LTL:
    def __init__(self, gw, model, **kwargs):
        self.gw = gw
        self.model = model
        self.visited_states_train = []
        self.visited_states_test = []
        self.LTL_coef = 10
        self.NN_value_active = False

        self.search_depth = kwargs['search_depth'] if 'search_depth' in kwargs else 100 
        self.MCTS_samples = kwargs['MCTS_samples'] if 'MCTS_samples' in kwargs else 100

        self.num_training_epochs = kwargs['num_training_epochs'] if 'num_training_epochs' in kwargs else 10
        self.num_test_epochs = kwargs['num_test_epochs'] if 'num_test_epochs' in kwargs else 20
        self.training = kwargs['training'] if 'training' in kwargs else True
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 10
        self.C = kwargs['C'] if 'C' in kwargs else 0.2
        self.tow = kwargs['tow'] if 'tow' in kwargs else 0.1
        self.T = kwargs['T'] if 'T' in kwargs else [50]
        self.K = kwargs['K'] if 'K' in kwargs else 1
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32
        self.steps_per_epoch = kwargs['steps_per_epoch'] if 'steps_per_epoch' in kwargs else 4
        self.best_val_len = {}
        for s in gw.csrl.states(): self.best_val_len[s] = (0.001, 99999)
        if os.path.exists("outputs/Log_run.txt"): os.remove("outputs/Log_run.txt")
    
    def train(self, num_training_epochs=None):
        if num_training_epochs != None: self.num_training_epochs = num_training_epochs
        idx = 0
        self.train_history = []

        for i in self.T:
            idx += 1
            print("##########################")
            print("C:",self.C, "| tow:",self.tow)
            # TRAIN ##############################
            train_wins = 0
            # num_training_epochs = int(200 - 1.9*i)
            # model = build_model(ch_states[(0,0,0,0)].shape, csrl.shape[-1])
            N, W, Q, P, visited_train = np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), set()
            for epoch in range(self.num_training_epochs):
                print(N[(0, 0, 4, 3)][:4])
                t1 = time.time()
                state_history, channeled_states, trajectory, action_history, reward_history, better_policy, best_val_len = MC_learning(
                    self.gw.csrl, self.model, self.gw.LTL_formula, self.gw.predicates, self.gw.csrl.reward, self.gw.ch_states,
                    N = N, W = W, Q = Q, P = P, C=self.C, tow=self.tow, n_samples=self.MCTS_samples, visited=visited_train,
                    start=None, search_depth=self.search_depth, verbose=0, T=i, K=self.K, NN_value_active=self.NN_value_active,
                    run_num=epoch, ltl_f_rew=True, reachability=True, best_val_len = self.best_val_len)
                
                self.visited_states_train += state_history
                t2 = time.time()
                # print(t2-t1, " run episode")

                # win = check_LTL(LTL_formula, trajectory, predicates)[0]
                if reward_history[-1]>0:
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
                # win_hist.append(win)
                t4 = time.time()
                # print(t4-t3, "fit", len(x_train))
            print("Train wins:",train_wins,"/", self.num_training_epochs)


    def test(self, num_test_epochs=None):
        if num_test_epochs != None: self.num_test_epochs = num_test_epochs
        for i in self.T:
            self.success_rates = []
            self.succes_std = []
            self.win_hist = []

            # TEST ##############################
            test_wins = 0
            N, W, Q, P, visited_test = np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), np.zeros(self.gw.csrl.shape), set()
            for epoch in range(self.num_test_epochs):
                print(N[(0, 0, 4, 3)][:4])
                state_history, channeled_states, trajectory, action_history, reward_history, better_policy, best_val_len = MC_learning(
                    self.gw.csrl, self.model, self.gw.LTL_formula, self.gw.predicates, self.gw.csrl.reward, self.gw.ch_states,
                    N = N, W = W, Q = Q, P = P, C=0.1, tow=0.05, n_samples=self.MCTS_samples, visited=visited_test, start=None,
                    search_depth=self.search_depth, verbose=0, T=i, K=1, NN_value_active=True, run_num=epoch, ltl_f_rew=True,
                    reachability=True, best_val_len = self.best_val_len)

                # win = check_LTL(LTL_formula, trajectory, predicates)[0]
                win = reward_history[-1]
                if win: test_wins+=1
                self.win_hist.append(win)
                self.visited_states_test += state_history
                
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
        # u, d, r, l