from Grid_world import *
from LTL import *
from NN import build_model

def make_move(grid_world, loc_i, loc_j, move, error_chance=0, right_chance=0.5, map_num=None):
  
  new_loc_i, new_loc_j = loc_i, loc_j
  
  if np.random.rand() < error_chance: # random move
    # determine if the random move is to right or left
    move_right = np.random.choice(2, 1, p=[right_chance, 1-right_chance]) == 0
    if move==4:
        pass
    elif move==0:
        if move_right: return make_move(grid_world, loc_i, loc_j, 3, error_chance=0)
        else: return make_move(grid_world, loc_i, loc_j, 1, error_chance=0)
    elif move==1:
        if move_right: return make_move(grid_world, loc_i, loc_j, 0, error_chance=0)
        else: return make_move(grid_world, loc_i, loc_j, 2, error_chance=0)
    elif move==2:
        if move_right: return make_move(grid_world, loc_i, loc_j, 1, error_chance=0)
        else: return make_move(grid_world, loc_i, loc_j, 3, error_chance=0)
    elif move==3:
        if move_right: return make_move(grid_world, loc_i, loc_j, 2, error_chance=0)
        else: return make_move(grid_world, loc_i, loc_j, 0, error_chance=0)
  else:
    if move==4:
        pass
    elif move==0 and loc_j< len(grid_world.map[0])-1 and grid_world.map[loc_i][loc_j+1]!=-3:
        new_loc_j += 1
    elif move==1 and loc_i > 0 and grid_world.map[loc_i-1][loc_j]!=-3:
        new_loc_i -= 1
    elif move==2 and loc_j > 0 and grid_world.map[loc_i][loc_j-1]!=-3:
        new_loc_j -= 1
    elif move==3 and loc_i < len(grid_world.map)-1 and grid_world.map[loc_i+1][loc_j]!=-3:
        new_loc_i += 1

  if map_num==None:
    new_grid_world = Grid_world(like=grid_world)
    new_grid_world.map[loc_i][loc_j] = 0

  else:
    new_grid_world = Grid_world(map_num=map_num)

  # apply the new location of the agent
  new_grid_world.map[new_loc_i][new_loc_j] = 1

  return new_grid_world, new_loc_i, new_loc_j

def hash_state(state):
  hash = ''
  for i in state.flatten():
    hash += str(i)
  return hash


def run_episode(grid_world, model, LTL_formula, predicates, n_steps=150, C=3, tow=1, n_samples=300, search_depth=None, N={}, W={}, Q={}, P={},
                random_move_chance=0, verbose=0, foo=0):
    state_history = []
    action_history = []
    better_policy = []
    trajectory = []
    reward = 0
    if search_depth == None: search_depth = n_steps + 1
    for step in range(n_steps):
        location_i, location_j, location = grid_world.get_locations(1, False)
        trajectory.append(location)

        ###### check if LTL specs are violated
        if location in predicates['d']:break

        state_history.append(grid_world.map.copy())
        # MCTS - policy improvment
        Pi = MCTS(model, hash_state(grid_world.map), {hash_state(grid_world.map):(grid_world, {})},
                  LTL_formula, predicates, trajectory[:-1], N, W, Q, P, n_samples=n_samples, depth=search_depth, tow=tow, C=C, foo=foo)
        better_policy.append(Pi.copy())
          
        # move based on enhanced policy
        # action = Pi.argmax() # greedy
        action = np.random.choice(5, p=Pi)
        action_history.append(action)
        
        # making the move in the grid world
        grid_world, _, _ = make_move(grid_world, location_i, location_j, action, error_chance=random_move_chance,
                                      map_num=grid_world.map_num)

        if verbose==1:
          print(action, end=", ")
        elif verbose==2:
          # clear_output(wait=True)
          print(grid_world.map, end="\r")
        elif verbose==3:
          print("step:",step, "MCTS Pi:",Pi)
          for i in N:
            print(i, N[i])

    outcome = check_LTL(LTL_formula, trajectory, predicates)
    if outcome[0]: reward = 1
    return state_history, action_history, reward, better_policy

def decayed_reward(size, init_rew, l=0.95):
  
  rew_seq = []

  for i in range(size):
    rew_seq.append(init_rew)
    init_rew *= l
  
  return np.array(rew_seq)

def run_simulation(grid_world_shape, n_goals, ordered_goals=False, n_episodes=10, C=3, tow=1, n_steps=100, n_samples=300, search_depth=None,
                    N={}, W={}, Q={}, P={}, verbose=True):

    grid_world_0 = Grid_world(shape=grid_world_shape, n_goals=n_goals, ordered=ordered_goals)
    if verbose: print(grid_world_0.map)
    model = build_model(grid_world_0.shape)
    goal_trajectory = grid_world_0.get_goal_trajectory()
    goal_locations = grid_world_0.get_locations(goal_trajectory, as_list=True)
    obstacle_locations = grid_world_0.get_locations(-1, as_list=True)
    LTL_formula = ('&', ('^', (None, 'g')), ('#', ('~', (None, 'o'))) )
    predicates = {
        'g':goal_locations,
        'o':obstacle_locations
    }

    for e in range(n_episodes):
        
        grid_world = grid_world_0.copy()
        
        state_history, action_history, reward, better_policy = run_episode(grid_world, model, LTL_formula, predicates, n_steps=n_steps,
                                                                            n_samples=n_samples, search_depth=search_depth, tow=tow, C=C, N=N, W=W, Q=Q, P=P, verbose=2)

        X = np.array(state_history).reshape(-1, grid_world.shape[0]*grid_world.shape[1])
        y1 = np.array(better_policy)
        # y2 = np.repeat(reward, len(better_policy))
        y2 = decayed_reward(size=len(better_policy), init_rew=reward, l=1)[::-1]
        model.fit(X, [y1, y2], epochs=50, verbose=0)

        # model.fit(np.array(state_history), np.array(better_policy), epochs=30, verbose=0) # without value output

        if verbose:
          # print(N[hash_state(grid_world)])
          # print(state_history)
          root = hash_state(state_history[-1])
          # print("N:",N[root])
          # print("Q:",Q[root])
          # print("U:", C * P[root] * (np.sqrt(np.sum(N[root])))/(1+N[root]))
          print("episode reward:",reward, ",model value estimate of last step", model(state_history[-1].reshape(1, -1))[1].numpy()[0][0])
          print("Action history:","(",len(action_history),")", action_history)
        if reward == 1: return True
    return False


############################## MCTS ##############################

# a_t = argmax(Q(s_t,a) +  U(s-t,a))

# NODE:
# N(s,a): visit count 
# W(s,a): total action-value
# Q(s,a): mean action-value
# P(s,a): prior probability
# U(s,a) = C * P(s,a) * (np.sqrt(np.sum(N(s,:))))/(1+N(s,a))

# leaf nodes:
# N(s_l, a) = 0
# W(s_l, a) = 0
# Q(s_l, a) = 0
# P(s_l, a) = P_a

# backward pass
# N(s,a) += 1
# W(s,a) += v
# Q(s,a) = W(s,a)/N(s,a)

# Pi(a|s_0) = N(s_0, a)**(1/tow) / np.sum(N(s_0,:)**(1/tow))

def MCTS_rec(model, root, tree, LTL_formula, trajectory, predicates, N={}, W={}, Q={}, P={}, C=1, depth=100, random_move_chance=0, foo=0):

  # tree structure: a dict from 'state_hash' -> (grid_world, children)
  grid_world = tree[root][0] # get the raw grid_world state
  location_i, location_j, location = grid_world.get_locations(target=1, absolute_loc=False) # get current location of the agent
  trajectory.append(location)

  ######## check if LTL specs are violated
  if location in predicates['d']: return -1

  if depth < 0: # search depth limit reached
      outcome = check_LTL(LTL_formula, trajectory, predicates)
      # if outcome[0]: print(trajectory)
      if outcome[0]: return 1
      else: return -1
  
  elif root not in N: # unexplored leaf node
      model_output = model(grid_world.map.reshape(1, -1))
      value = model_output[1].numpy()[0][0]
      P[root] = model_output[0].numpy()[0]
      N[root] = np.zeros(5)
      W[root] = np.zeros(5)
      Q[root] = np.zeros(5)
      return value

  ### selecting the next node to expand ###
  U = C * P[root] * (np.sqrt(np.sum(N[root])))/(1+N[root])
  next_move = (U + Q[root]).argmax()
  # next_move = np.random.randint(4)
  #########################################
  
  ### creating the next subtree ###
  next_grid_world, new_location_i, new_location_j = make_move(grid_world, location_i, location_j, next_move, error_chance=random_move_chance,
                                      map_num=grid_world.map_num)
  sub_root = hash_state(next_grid_world.map)
  sub_tree = {sub_root :(next_grid_world, {})}
  #################################
  ### expanding the next move and back tracking ###
  value = MCTS_rec(model, sub_root, sub_tree, LTL_formula, trajectory, predicates,  N, W, Q, P, C=C, depth=depth-1, foo=foo)
  N[root][next_move] += 1
  # if root == '01-100022-30-1020200-10-1':
  #   print(U + Q[root])
  #   print("!!!", sub_root==root, len(trajectory))
  W[root][next_move] += value
  # if root == '1-10-120-10000-10-10000-100-1000' and next_move==3:
    # print("value:", value)
    # print('U:',U,'Q:', Q[root])
  Q[root][next_move] = W[root][next_move]/N[root][next_move]
  #################################################
  return value

def MCTS(model, root, tree, LTL_formula, predicates, trajectory, N, W, Q, P, n_samples=100, tow=1, C=1, depth=100, foo=0):

  grid_world = tree[root][0] # get the grid_world
  
  for sample in range(n_samples):
    MCTS_rec(model, root, tree.copy(), LTL_formula, trajectory.copy(), predicates,  N, W, Q, P, C=C, depth=depth-len(trajectory), foo=foo)

  Pi = (N[root]**(1/tow)) / np.sum(N[root]**(1/tow))
  return Pi

# g = Grid_world(shape=(5,5), n_goals=1)
# make_move(g, 0, 0, 1, error_chance=0, right_chance=0.5)
