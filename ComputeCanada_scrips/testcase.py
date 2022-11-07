from Grid_world import *
from NN import *
from EnvSim import run_episode, hash_state, decayed_reward, MCTS

import time
from matplotlib import pyplot as plt
from collections import Counter

t1 = time.time()
"""Test case for the first case of the paper"""

def run_simulation(grid_world_shape, n_goals, formula, predicates, model, ordered_goals=False,
                   n_episodes=10, C=3, tow=1, n_steps=100, n_samples=300, N={}, W={}, Q={}, P={},
                   random_move_chance=0, verbose=True, training=True):

    grid_world_0 = Grid_world(map_num=grid_world_shape, n_goals=n_goals, ordered=ordered_goals, rand_init=True)
    grid_world_0.add_item(1, 1) # adding the init location randomly
    # grid_world_0.map[0,0] = 1
    if verbose: print(grid_world_0.map)

    x_train = grid_world_0.map.reshape(-1, grid_world_0.shape[0]*grid_world_0.shape[1])
    y1_train = 0.2*np.ones((1,5))
    y2_train = np.array([1])
    rewards= []

    for e in range(n_episodes):
        
        grid_world = grid_world_0.copy()
        
        state_history, action_history, reward, better_policy = run_episode(grid_world, model, formula, predicates,
                                                n_steps=n_steps, n_samples=n_samples, tow=tow, C=C, N=N, W=W, Q=Q, P=P,
                                                random_move_chance = random_move_chance, verbose=0)
        last_position = (np.where(state_history[-1]==1)[0][0]*4 + np.where(state_history[-1]==1)[1][0])

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
        
        rewards.append(reward)
    return np.sum(rewards), last_position    

predicates={'a':[6,14], 'b':[7,12], 'c':[2,10,17,19]}

formula1 = ('[]', ('~', (None, 'c')))
formula22 = ('<>', ('[]', (None, 'a')))
formula32 = ('<>', ('[]', (None, 'b')))
formula4 = ('\\/', formula22, formula32)
formula = ('/\\', formula1, formula4)

model = build_model((5,4))

print("Deterministic Env, Training initiated...")

# agent training
train_ress = 0
train_locs = []
n_games=50
n_episodes=2
N, W, Q, P = {}, {}, {}, {}
for i in range(n_games):
  res, loc = run_simulation(6, n_goals=1, formula=formula,predicates=predicates,model=model,
                            n_episodes=n_episodes, C=1, tow=0.1, n_steps=8, n_samples=400,random_move_chance=0,
                            N=N, W=W, Q=Q, P=P, verbose=True, training=True)
  train_ress += res
  train_locs.append(loc)

  if i % 50 == 0:
    print("i=",i,"| total train_ress:",train_ress,"/",n_games*n_episodes)


print("#######################################")
print("total train_ress:",train_ress,"/",n_games*n_episodes)

start_pos = [(i,j) for i in range(5) for j in range(4)]
values = []
policies = []
ps = []
for s_p in start_pos:
  grid_world_0 = Grid_world(map_num=6, n_goals=1)
  grid_world_0.map[s_p[0],s_p[1]]=1 # adding the init location randomly
  print(grid_world_0.map)

  model_output = model(grid_world_0.map.reshape(1, -1))
  value = model_output[1].numpy()[0][0]
  policy = model_output[0].numpy()[0]
  try:
    Pi = MCTS(model, hash_state(grid_world_0.map), {hash_state(grid_world_0.map):(grid_world_0, {})},
              formula, predicates, N, W, Q, P, n_samples=400, tow=0.1, C=1, depth=6)
  except:
    values.append(value)
    ps.append(np.argmax(policy))
    policies.append(5)
    continue
  values.append(value)
  ps.append(np.argmax(policy))
  policies.append(np.argmax(Pi))

print(np.reshape(policies, (5, 4)))
print(np.reshape(ps, (5, -1)))

# agent testing
test_ress = []
test_locs = []
for i in range(10):
  N, W, Q, P = {}, {}, {}, {}
  res, loc = run_simulation(6, n_goals=1, formula=formula,predicates=predicates,model=model,
                            n_episodes=1, C=1, tow=0.1, n_steps=10, n_samples=400,random_move_chance=0,
                            N=N, W=W, Q=Q, P=P, verbose=True, training=False)
  test_ress.append(res)
  test_locs.append(loc)

print("#######################################")
print("Counter(test_ress):",Counter(test_ress))
print("Counter(test_locs):",Counter(test_locs))

print("took",time.time()-t1,"seconds")