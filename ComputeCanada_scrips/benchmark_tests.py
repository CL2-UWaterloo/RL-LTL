from Grid_world import Grid_world
from NN import build_model
from EnvSim import run_episode, run_simulation, hash_state
from matplotlib import pyplot as plt

"""Test cases for end at goal position"""

shapes = [(i,i) for i in range(5,25,2)]
results_1 = []

for gw_shape in shapes:
  print(gw_shape, end=": ")
  for _ in range(5):
    N, W, Q, P = {}, {}, {}, {}
    outcome = run_simulation(gw_shape, 2, n_episodes=3, C=1, tow=0.1, n_steps=gw_shape[0]*2+5, n_samples=500, N=N, W=W, Q=Q, P=P, verbose=False)
    results_1.append(outcome)
    print(outcome,end=", ")
  print()

"""same as above, but also with goal trajectory"""

np.reshape(results_1[:55],(-1,5)), results_1[55:]

shapes = [(i,i) for i in range(5,40,2)]
results_2 = []
# 2 Goals
for gw_shape in shapes:
  print(gw_shape, end=": ")
  for _ in range(5):
    N, W, Q, P = {}, {}, {}, {}
    outcome = run_simulation(gw_shape, 2, ordered_goals=True, n_episodes=3, C=1, tow=0.1, n_steps=gw_shape[0]*2+5, n_samples=500, N=N, W=W, Q=Q, P=P, verbose=False)
    results_2.append(outcome)
    print(outcome,end=", ")
  print()

results_3 = []
# 3 Goals
for gw_shape in shapes:
  print(gw_shape, end=": ")
  for _ in range(5):
    N, W, Q, P = {}, {}, {}, {}
    outcome = run_simulation(gw_shape, 3, ordered_goals=True, n_episodes=3, C=1, tow=0.1, n_steps=gw_shape[0]*2+5, n_samples=500, N=N, W=W, Q=Q, P=P, verbose=False)
    results_3.append(outcome)
    print(outcome,end=", ")
  print()

"""test cases for multiple goal reach (trajectory)"""

shapes = [(i,i) for i in range(5,38, 2)]
results_1 = []
print("1 goal")
for gw_shape in shapes:
  print(gw_shape, end=": ")
  for _ in range(5):
    N, W, Q, P = {}, {}, {}, {}
    outcome = run_simulation(gw_shape, 1, n_episodes=3, C=1, tow=0.1, n_steps=gw_shape[0]*2+5, n_samples=500, N=N, W=W, Q=Q, P=P, verbose=False)
    results_1.append(outcome)
    print(outcome,end=", ")
  print()

print("2 goals")
results_2 = []
for gw_shape in shapes:
  print(gw_shape, end=": ")
  for _ in range(5):
    N, W, Q, P = {}, {}, {}, {}
    outcome = run_simulation(gw_shape, 2, n_episodes=3, C=1, tow=0.1, n_steps=gw_shape[0]*2+5, n_samples=500, N=N, W=W, Q=Q, P=P, verbose=False)
    results_2.append(outcome)
    print(outcome,end=", ")
  print()

print("3 goals")
results_3 = []
for gw_shape in shapes:
  print(gw_shape, end=": ")
  for _ in range(5):
    N, W, Q, P = {}, {}, {}, {}
    outcome = run_simulation(gw_shape, 3, n_episodes=3, C=1, tow=0.1, n_steps=gw_shape[0]*2+5, n_samples=500, N=N, W=W, Q=Q, P=P, verbose=False)
    results_3.append(outcome)
    print(outcome,end=", ")
  print()

print("3 goals ordered")
results_3_1 = []
for gw_shape in shapes:
  print(gw_shape, end=": ")
  for _ in range(5):
    N, W, Q, P = {}, {}, {}, {}
    outcome = run_simulation(gw_shape, n_goals= 3, ordered_goals=True, n_episodes=3, C=1, tow=0.1, n_steps=gw_shape[0]*2+5, n_samples=500, N=N, W=W, Q=Q, P=P, verbose=False)
    results_3_1.append(outcome)
    print(outcome,end=", ")
  print()

plt.bar(range(5,27,2), np.reshape(results_1[:55], (-1, 5)).mean(1))
plt.title("success rate based on grid world size - stay at end goal (1 goal)")
plt.xlabel("dim of grid_wrold")
plt.ylabel("success rate")
plt.figure()

plt.bar(range(5,38,2), np.reshape(results_2, (-1, 5)).mean(1))
plt.title("success rate based on levels of complexity (2 goals)")
plt.xlabel("dim of grid_wrold")
plt.ylabel("success rate")

plt.bar(range(5,38,2), np.reshape(results_3, (-1, 5)).mean(1))
plt.title("success rate based on levels of complexity (3 goals)")
plt.xlabel("dim of grid_wrold")
plt.ylabel("success rate");

plt.bar(range(5,38,2), np.reshape(results_3_2, (-1, 5)).mean(1))
plt.title("success rate based on levels of complexity (3 ordered goals)")
plt.xlabel("dim of grid_wrold")
plt.ylabel("success rate");

# np.reshape(results, (-1, 10)).mean(1)
# plt.plot(results)
fin_res = np.ones(30)
fin_res[5:25] = res.mean(1)
fin_res[25:] = np.reshape(results, (-1, 10)).mean(1)
fin_res

# res = np.reshape(results, (20, 10))
res.mean(1)
plt.bar(range(5,25), res.mean(1))

# grid_world = reset_grid_world(21)
grid_world, hash_state(grid_world)

for i in N:
  print(i, N[i])

t = np.array([0,0,0,0,0])
model(t.reshape(1, -1))[0][0].numpy().argmax()

grid_world= reset_grid_world()
state_history, action_history, reward, possible_policy = run_episode(grid_world.copy(), model, eps=0.1)

def show_movement(states):
  grid_world = reset_grid_world()
  goal_location = get_locations(grid_world, 2)
  obstacle_locations = get_locations(grid_world, -1)
  for state in states:
    i, j = int(state[0]//5), int(state[0]%5)
    grid_world[i,j] = 1
    print(grid_world)
    grid_world[i][j] = 0
    print()

print(model(np.reshape([0, -1, -1, -1, 0], (1, -1)))[0].numpy(), model(np.reshape([0, -1, -1, -1, 0], (1, -1)))[0].numpy().argmax())

show_movement(state_history)

# PROBLEMS:

# infinite loops
# too long tragectories, agent can't find the way
# tree search helps it NAVIGATE, but it usually gets stuck

# effect of positive/negative rewards on the path taken
    # use value functions?  
    # intermediate rewards?
    # deep MCTS?

# * turn this into a simple baseline to try out different methods (ranked rewards,...)


# hard core results of previous tests (kept just in case):
# 1 goal
results_1=[
True, True, True, True, True, 
True, True, True, True, True, 
True, True, True, True, True, 
True, True, True, True, True, 
True, False, True, True, True, 
False, True, True, True, True, 
False, True, True, True, True, 
False, True, True, True, True, 
False, True, True, False, True, 
False, True, True, True, True, 
True, False, True, True, False, 
True, False, True, True, False, 
True, True, True, True, True, 
True, True, True, True, True, 
True, False, False, False, True, 
True, True, True, False, True, 
False, True, True, False, False]

# 2 goals
results_2=[
True, True, True, True, True, 
True, True, True, True, True, 
True, True, True, True, True, 
True, True, True, True, True, 
True, False, True, True, False, 
True, True, False, False, True, 
True, True, True, True, True, 
False, True, True, True, False, 
True, False, True, False, True, 
True, True, True, True, True, 
False, True, False, True, False, 
False, False, True, False, True, 
False, False, False, True, True, 
False, False, True, False, False, 
True, True, False, True, False, 
False, False, False, False, True,
False, False, False, False, False]

# 3 goals
results_3=[
True, True, True, True, True, 
True, True, True, True, True, 
True, True, True, True, True, 
True, True, True, False, True, 
True, False, True, True, True, 
True, True, False, False, True, 
True, True, True, True, True, 
False, True, True, True, False, 
True, False, False, True, True, 
True, False, False, False, True, 
True, True, True, True, False, 
False, False, True, False, True, 
False, False, False, True, True, 
False, True, False, False, False, 
False, False, True, True, False, 
False, False, False, True, False,
False, False, False, False, True]

# ordered goals 
results_3_2=[
True, False, False, True, True, 
True, True, True, True, True, 
True, False, False, True, True, 
True, True, False, False, True, 
True, False, True, True, True, 
False, True, False, False, True, 
False, False, False, True, True, 
False, True, True, True, False, 
True, False, False, False, True, 
False, False, False, False, True, 
False, True, False, True, False, 
False, False, False, True, False, 
True, False, False, False, False, 
False, False, False, False, False, 
True, False, False, False, False, 
False, False, True, False, False,
False, False, False, False, False]

# stay at end goal results_1 
array([[ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True, False],
        [ True,  True,  True, False,  True],
        [ True, False,  True, False,  True]]), [True, True])

# stay at end goals (2)
(array([[ True,  True, False,  True, False],
        [False,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [False,  True, False,  True, False],
        [ True, False, False, False, False],
        [False,  True, False, False, False],
        [False, False,  True,  True, False],
        [False, False, False,  True, False],
        [False, False,  True, False,  True],
        [False, False, False, False,  True]]),
 [False, False, False])

# stay at end goals (3)
array([[False, False, False,  True, False],
        [ True,  True, False, False, False],
        [False,  True, False,  True, False],
        [False, False, False, False, False],
        [False, False,  True, False, False],
        [False, False,  True,  True, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]]),
