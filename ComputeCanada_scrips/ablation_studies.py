from Grid_world import *
from NN import *
from LTL import *
from EnvSim import run_episode, hash_state, decayed_reward

import time
from matplotlib import pyplot as plt
from collections import Counter

def run_simulation(map_num, n_goals, formula, predicates, model, ordered_goals=False,
                   n_episodes=10, C=3, tow=1, n_steps=100, n_samples=100, search_depth=None, N={}, W={}, Q={}, P={},
                   random_move_chance=0, verbose=True, training=True, foo=0):

    grid_world_0 = Grid_world(map_num=map_num, n_goals=1, ordered=False, rand_init=False)
    grid_world_0.map[0,0] = 1

    if verbose:
        print(grid_world_0.map)
        # print(len(N),len(W), len(Q), len(P))

    x_train = grid_world_0.map.reshape(-1, grid_world_0.shape[0]*grid_world_0.shape[1])
    y1_train = 0.2*np.ones((1,5))
    y2_train = np.array([1])
    rewards= []

    for e in range(n_episodes):
        grid_world = grid_world_0.copy()
        N.clear(), W.clear(), Q.clear(), P.clear()
        print(N)
        state_history, action_history, reward, better_policy = run_episode(grid_world, model, formula, predicates,
                                                n_steps=n_steps, n_samples=n_samples, search_depth=search_depth,
                                                tow=tow, C=C, N=N, W=W, Q=Q, P=P,
                                                random_move_chance = random_move_chance, verbose=0, foo=foo)
        last_position = (np.where(state_history[-1]==1)[0][0]*4 + np.where(state_history[-1]==1)[1][0])
        # for c,i in enumerate(N):
        #       print(i, N[i])
        if training:
          X = np.array(state_history).reshape(-1, grid_world.shape[0]*grid_world.shape[1])
          x_train = np.append(x_train, X, axis=0)
          y1 = np.array(better_policy)
          y1_train = np.append(y1_train, y1, axis=0)
          y2 = decayed_reward(size=len(better_policy), init_rew=reward, l=1)[::-1]
          y2_train = np.append(y2_train, y2)

          model.fit(x_train, [y1_train, y2_train], epochs=20, verbose=0)

        if verbose:
          root = hash_state(state_history[-1])
          print("episode reward:",reward, ", value estimate of last step", model(state_history[-1].reshape(1, -1))[1].numpy()[0][0])
          print("Action history:","(",len(action_history),")", action_history)
        
        rewards.append(reward)
        
    # for c,i in enumerate(N):
    #   print(i, N[i])
    #   if c>4: break
    return np.sum(rewards), last_position

grid_world_0 = Grid_world(map_num=6, n_goals=1, ordered=False, rand_init=False)
goal_locations = grid_world_0.get_locations(2, as_list=True)
obstacle_locations = grid_world_0.get_locations(-1, as_list=True)

print("Ablation Studies on MCTS depth | Trying with trainable NN...")
model = build_model(grid_world_0.shape)

predicates={'a':[6,14], 'b':[7, 12], 'd':obstacle_locations}
# predicates={'a':goal_locations, 'd':obstacle_locations}

formula = parser.parse("(<> [] a \/ <> [] b) /\ [] ~ d")
# formula = parser.parse("<> [] a /\ [] ~ d")
print(formula)
print(predicates)

# agent training
train_ress = 0
exp_results = []
MCTS_samples = range(1,10)
n_episodes=10

for i, MCTS_s in enumerate(MCTS_samples):
    res, loc = run_simulation(6, n_goals=1, formula=formula, predicates=predicates, model=model,
                            n_episodes=n_episodes, C=1, tow=0.1, n_steps=50, n_samples=10, search_depth=MCTS_s, random_move_chance=0,
                            N={}, W={}, Q={}, P={}, verbose=True, training=True, foo=0)
    train_ress += res
    exp_results.append(res*(100//n_episodes))
    print("samples:", MCTS_s, "success:", res,"/",n_episodes,"\n\n\n")

plt.plot(MCTS_samples, exp_results)
plt.title("Det env, time-steps:100, #samples:10, training: ON")
plt.ylabel("Success rate (%)")
plt.xlabel("search depth of MCTS")
plt.show()

# res, loc = run_simulation(2, n_goals=1, formula=formula, predicates=predicates, model=model,
#                             n_episodes=n_episodes, C=1, tow=0.1, n_steps=15, n_samples=50, search_depth=3, random_move_chance=0,
#                             N={}, W={}, Q={}, P={}, verbose=True, training=False)