from Grid_world import *
from NN import *
from LTL import *
from EnvSim import run_episode, hash_state, decayed_reward

import time
from matplotlib import pyplot as plt
from collections import Counter
import argparse

# Initialize parser
argparser = argparse.ArgumentParser()

# Adding optional argument
argparser.add_argument("-t", "--timesteps", default=30,type=int, help = "# time-steps of the MCTS")
argparser.add_argument("-s", "--samples", default=30, type=int, help = "# samples of the MCTS")
argparser.add_argument("-d", "--depth", default=30, type=int, help = "depth of the MCTS")
argparser.add_argument("-n", "--nruns", default=20, type=int, help = "# (independent) runs of the simulation")
argparser.add_argument("-e", "--episodes", default=2, type=int, help = "# episodes per run of the MCTS")
argparser.add_argument("-r", "--randomness", default=0, type=float, help = "Amount of randomness of the env")
argparser.add_argument("-l", "--LTLformula", help = "the LTL formula for evaluation")

# Read arguments from command line
args = argparser.parse_args()

def run_simulation(grid_world_shape, n_goals, formula, predicates, model, ordered_goals=False,
                   n_episodes=10, C=3, tow=1, n_steps=100, n_samples=300, N={}, W={}, Q={}, P={},
                   random_move_chance=0, verbose=True, training=True):

    grid_world_0 = Grid_world(map_num=grid_world_shape, n_goals=n_goals, ordered=ordered_goals, rand_init=True)
    # grid_world_0.add_item(1, 1) # adding the init location randomly
    grid_world_0.map[4,1] = 1
    if verbose: print(grid_world_0.map)

    x_train = grid_world_0.map.reshape(-1, grid_world_0.shape[0]*grid_world_0.shape[1])
    y1_train = 0.2*np.ones((1,5))
    y2_train = np.array([1])
    rewards= []

    for e in range(n_episodes):
        grid_world = Grid_world(like=grid_world_0)
        
        state_history, action_history, reward, better_policy = run_episode(grid_world, model, formula, predicates,
                                                n_steps=n_steps, n_samples=n_samples, tow=tow, C=C, N=N, W=W, Q=Q, P=P,
                                                random_move_chance = random_move_chance, verbose=2)
        trajectory_history = [np.where(s==1)[0][0]*4 + np.where(s==1)[1][0] for s in state_history]
        last_position = trajectory_history[-1]

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
          print("episode reward:",reward, ",model value estimate of last step", model(state_history[-1].reshape(1, -1))[1].numpy()[0][0])
          print("Action history:","(",len(action_history),")", action_history)
          print("Trajectory:", trajectory_history)

        for c,i in enumerate(N):
          print(i, N[i])
          if c>3: break    
        rewards.append(reward)
    
    return np.sum(rewards), last_position    

predicates={'a':[12], 'b':[2], 'c':[17], 'd':[3]}

# Parse the LTL expression
full_t = "[] ( (~d) /\ ((b /\ ~ > b) -> >(~b % (a \/ c))) /\ (a -> >(~a % b))"
full_t += " /\ ((~b /\ >b /\ ~>>b)->(~a % c)) /\ (c->(~a % b)) /\ ((b /\ >b) -> <>a))"
if args.LTLformula == None:
  t = "[] ( (~d) /\ (c->(~a % b)) )"
else:
  t = args.LTLformula
#    [] ( (~d) /\ (c->(~a % b)) /\ ((b /\ >b) -> <>a) /\ ((b /\ ~>b) -> >(~b % (a \/ c))) /\ (a -> >(~a % b))
#  /\ ((~b /\ >b /\ ~>>b)->(~a % c)) /\ (c->(~a % b)) )

t = "[] ( (~d) /\ (c->(~a % b)) /\ ((b /\ >b) -> <>a) /\ ((b /\ ~ > b) -> >(~b % (a \/ c))) )"

formula = parser.parse(t)
print(formula)

model = build_model((5,4))
tra = [17, 13, 9, 5, 6, 7, 6, 7, 2, 3]
print(check_LTL(formula, tra, predicates))

print("Deterministic Env, Training initiated...")
print(args)

# agent training
train_ress = 0
exp_results = []
n_games= range(args.nruns)
n_episodes = args.episodes

for i in n_games:
  N, W, Q, P = {}, {}, {}, {}
  res, loc = run_simulation(7, n_goals=1, formula=formula,predicates=predicates,model=model,
                            n_episodes=n_episodes, C=1, tow=0.1, n_steps=args.timesteps, n_samples=args.samples,
                            random_move_chance=args.randomness, N=N, W=W, Q=Q, P=P, verbose=True, training=True)
  train_ress += res
  exp_results.append(res*(100//n_episodes))
  print("___________________")
  if i % 49 == 0:
    print("i=",i,"| total train_ress:",train_ress,"/",(i+1)*n_episodes)

print("#######################################")
print("total train_ress:",train_ress,"/",len(n_games)*n_episodes)
print(np.mean(exp_results))

exit()

# agent testing
test_ress = []
test_locs = []
for i in range(10):
  res, loc = run_simulation(7, n_goals=1, formula=formula,predicates=predicates,model=model,
                            n_episodes=1, C=1, tow=0.1, n_steps=10, n_samples=50,random_move_chance=0,
                            N={}, W={}, Q={}, P={}, verbose=True, training=False)
  test_ress.append(res)
  test_locs.append(loc)

print("#######################################")
print("Counter(test_ress):",Counter(test_ress))
print("Counter(test_locs):",Counter(test_locs))

print("took",time.time()-t1,"seconds")