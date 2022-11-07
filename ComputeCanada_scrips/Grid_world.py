import numpy as np

class Grid_world:
  def __init__(self, map_num=None, shape=None, n_obstacles=None, n_goals=None, like=None, ordered=False, rand_init=False):
    
    self.shape = shape
    self.map_num = map_num
    self.map = None
    self.ordered = ordered
    self.n_goals = n_goals
    self.n_obstacles = n_obstacles
    
    if like != None:
      self.shape = like.shape
      self.map = like.map.copy()
      self.n_goals = like.n_goals
      self.n_obstacles = like.n_obstacles
      self.ordered = like.ordered

    elif map_num != None:
      self.map = self.set_grid_world(map_num)
      self.shape = self.map.shape
    elif shape != None:
      self.generate_grid_world(self.shape, n_obstacles, n_goals, rand_init)
    else:
      print("map_num & shape can't be 'None' at the same time.")

  def set_grid_world(self, map):
        if map == 0:
          grid_world = np.zeros((5,5))

          grid_world[0,4] = 2 # destination
          grid_world[0,0] = 1 # starting point

        elif map == 1:

          grid_world = np.zeros((5,5))
          grid_world[[2,3],3] = -1 # holes
          grid_world[[0,1],1] = -1 # holes

          grid_world[0, 4] = 2 # destination
          grid_world[0,0] = 1 # starting point

        elif map == 2:

          grid_world = np.zeros((5,5))
          grid_world[[0,2,3],3] = -1 # holes
          grid_world[[0,1,2,4],1] = -1 # holes

          grid_world[0, 4] = 2 # destination
          grid_world[0,0] = 1 # starting point

        elif map == 3:

          grid_world = np.zeros((10,10))
          grid_world[[0,6,7,9],1] = -1 # holes
          grid_world[[0,1,2,3,8],3] = -1 # holes
          grid_world[[0,1,2,3,4,5],5] = -1 # holes

          grid_world[-1, -1] = 2 # destination
          grid_world[0,0] = 1 # starting point

        elif map == 4:

          grid_world = np.zeros((10,10))
          grid_world[[0,1,2,4,6,7,9],1] = -1 # holes
          grid_world[[0,2,3, 6,7,8],3] = -1 # holes
          grid_world[[1,2,3,4,5,6],5] = -1 # holes
          grid_world[[2,3,5,6,7,9],7] = -1 # holes

          grid_world[-1, -1] = 2 # destination
          grid_world[0,0] = 1 # starting point
        
        elif map == 5:

          grid_world = np.zeros((16,16))
          grid_world[[0,1,2,4,6,7,9],1] = -1 # holes
          grid_world[[0,2,3, 6,7,8],3] = -1 # holes
          grid_world[[1,2,3,4,5,6],5] = -1 # holes
          grid_world[[2,3,5,6,7,9],7] = -1 # holes
          grid_world[[4,5,6,7,8,9,10,11,12,13,14,15],9] = -1 # holes
          grid_world[[1,2,3,4,5,6,7,8, 12, 14],11] = -1 # holes
          grid_world[[7,9,10,11,12,13,15],13] = -1 # holes

          grid_world[-2, -3] = 2 # destination
          grid_world[0,0] = 1 # starting point
        
        elif map == 6: # First case study of The "Control Synthesis from LTL Specifications using RL" Paper

          grid_world = np.zeros((5,4))
          grid_world[[0,2],2] = -1 # holes
          grid_world[4,[1,3]] = -1 # holes
          grid_world[2,0] = -3 # Obsticle

          grid_world[1, [2,3]] = 2 # destination
          grid_world[3, [0,2]] = 2 # destination
          
          # will add starting point later, randomly

        elif map == 7: # second case study of the "Control Synthesis from LTL Specifications using RL" Paper

          grid_world = np.zeros((5,4))

          grid_world[0, 2] = 2 # b
          grid_world[0, 3] = 3 # d
          grid_world[3, 0] = 4 # a
          grid_world[4, 1] = 5 # c
          
          # will add starting point later, randomly

        elif map == 10:

          grid_world = np.zeros((3,3))
          grid_world[1,[0,1]] = -1 # holes

          grid_world[-1, 0] = 2 # destination
          grid_world[0,0] = 1 # starting point

        elif map == 20:

          grid_world = np.zeros((8,8))

          grid_world[[0,-1], -1] = 2 # destinations
          grid_world[0,0] = 1 # starting point

        elif map == 21:

          grid_world = np.zeros((5,5))

          grid_world[[0,-1], -1] = 2 # destinations
          grid_world[0,0] = 1 # starting point

          grid_world[[0,2,3],3] = -1 # holes
          grid_world[[0,1,2,4],1] = -1 # holes
        
        elif map == 22:

          grid_world = np.zeros((5,5))

          grid_world[[0,-1], -1] = 2 # destinations
          grid_world[[0,-1], -3] = 2 # destinations
          grid_world[0,0] = 1 # starting point

          grid_world[[0,2,3],3] = -1 # holes
          grid_world[[0,1,2,4],1] = -1 # holes

        elif map == 24:

          grid_world = np.zeros((10,10))

          grid_world[[0,-1], -1] = 2 # destinations
          grid_world[0,0] = 1 # starting point
          
          grid_world[[0,6,7,9],1] = -1 # holes
          grid_world[[0,1,2,3,8],3] = -1 # holes
          grid_world[[0,1,2,3,4,5],5] = -1 # holes

        elif map == 25:

          grid_world = np.zeros((16,16))

          grid_world[[0,-1], -1] = 2 # destinations
          grid_world[[2,-3], -4] = 2 # destinations
          grid_world[0,0] = 1 # starting point
          
          grid_world[[0,6,7,9],1] = -1 # holes
          grid_world[[0,1,2,3,8],3] = -1 # holes
          grid_world[[0,1,2,3,4,5],5] = -1 # holes

        elif map == 30:

          grid_world = np.zeros((10,10))

          grid_world[0, -1] = 2 # destinations
          grid_world[1, -2] = 3 # destinations
          grid_world[2, -3] = 4 # destinations
          grid_world[0,0] = 1 # starting point
          
          grid_world[[0,6,7,9],1] = -1 # holes
          grid_world[[0,1,2,3,8],3] = -1 # holes
          grid_world[[0,1,2,3,4,5],5] = -1 # holes

        return grid_world.astype(int)

  def get_locations(self, target, absolute_loc=True, as_list=False):

      if target==None: i, j = np.where(self.map==2)
      elif type(target)==int: i, j = np.where(self.map==target)
      else:
        i, j = [], []
        for t in target:
          x,y = np.where(self.map==t)
          try:
              i.append(x[0])
              j.append(y[0])
          except:
            pass # the goal was overwritten by the agent, ignore it

      if absolute_loc:
        if len(i)>1 or as_list: return [(i[k]*self.shape[1] + j[k]) for k in range(len(i))]
        else: return i[0]*self.shape[1] + j[0]

      else:
        if len(i)>1 or as_list: return [(i[k], j[k], i[k]*self.shape[1] + j[k]) for k in range(len(i))]
        else: return i[0], j[0], i[0]*self.shape[1]+ j[0]
        
  def get_surroundings(self, location_i, location_j):

      surroundings = -np.ones(4)
      if location_j < len(self.map)-1:
          surroundings[0] = self.map[location_i][location_j+1]
      if location_i > 0:
          surroundings[1] = self.map[location_i-1][location_j]
      if location_j > 0:
          surroundings[2] = self.map[location_i][location_j-1]
      if location_i < len(self.map)-1:
          surroundings[3] = self.map[location_i+1][location_j]
      return surroundings

  def add_item(self, n_items, item):

      x,y = self.shape
      for _ in range(n_items):
        i,j = np.random.randint(x), np.random.randint(y)
        while(self.map[i,j]!=0):
          i,j = np.random.randint(x), np.random.randint(y)
        self.map[i,j] = item
        if self.ordered: item += 1

  def add_obstacles(self, n_obstacles=None):

      x,y = self.shape

      obstacles_in_row = x//2

      k = 0
      for j in range(1,y,2):
        for k in range(obstacles_in_row):
          i = np.random.randint(x)
          if self.map[i,j]==0:
            self.map[i,j] = -1
        
  def generate_grid_world(self, shape, n_obstacles, n_goals, rand_init):

    self.map = np.zeros(shape)
    
    # adding starting point
    if rand_init:
      self.add_item(1, 1)
    else:
      self.map[0,0] = 1

    self.add_item(n_goals, 2) # adding goals

    self.add_obstacles(n_obstacles)

  def get_goal_trajectory(self, offset=2):
    if self.ordered: return [i+offset for i in range(self.n_goals)]
    # else: return [offset for i in range(self.n_goals)]
    else: return None
  
  def copy(self):
    return Grid_world(like=self)
