from dependencies.LTL import *
from dependencies.Utility_funcs import *

from dependencies.csrl.mdp import GridMDP
from dependencies.csrl.oa import OmegaAutomaton
from dependencies.csrl import ControlSynthesis
import numpy as np

class grid_world:
    def __init__(self, name = 'random', p=1, plot=True, n_danger=None, shape=(5,5)):
        self.name = name
        self.plot = plot
        self.p = p
        self.n_danger = n_danger
        if name == 'random': self.generate_gw(shape)
        else:
            self.ltl = self.get_ltl(self.name)
            self.mdp = self.get_mdp(self.name, self.p)
            self.csrl = self.build(self.mdp, self.ltl, plot=plot)
            self.set_ltl_f()

    def build(self, grid_mdp, ltl, plot=False):
        # Translate the LTL formula to an LDBA
        oa = OmegaAutomaton(ltl)
        print('Number of Omega-automaton states (including the trap state):',oa.shape[1])

        if plot: grid_mdp.plot(save="env.pdf")

        # Construct the product MDP
        csrl = ControlSynthesis(grid_mdp,oa)
        self.max_rew = round(csrl.reward.max(), 3)

        self.s_vectors = state_vectors(csrl)
        enc = list(np.unique(grid_mdp.label))
        enc.pop(enc.index(()))
        self.ch_states = channeled(csrl, enc)
        self.total_number_of_states = csrl.mdp.shape[0]*csrl.mdp.shape[1]*csrl.oa.shape[1]

        return csrl

    def get_mdp(self, name, p):
        if name == "sequential_delivery":
            # MDP Description
            shape = (5,5)
            # E: Empty, T: Trap, B: Obstacle
            structure = np.array([
            ['E',  'E',  'E',  'E',  'E'],
            ['E',  'B',  'E',  'B',  'E'],
            ['E',  'E',  'E',  'E',  'E'],
            ['T',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E']
            ])

            # Labels of the states
            label = np.array([
            [(),    (),('a',),    (),    ()],
            [(),    (),    (),    (),    ()],
            [(),    (),    (),    (),    ()],
            [('c',),    (),    (),    (),    ()],
            [(),(),    ('b',),(),    ()]
            ],dtype=object)
            d_idx = []
            if self.n_danger is None: d_idx = [(1,2), (4,0), (4,1), (3,1), (3,2), (3,3)]
            elif self.n_danger == 1: d_idx = [(4,0)]
            elif self.n_danger == 2: d_idx = [(4,0), (4,1)]
            elif self.n_danger == 3: d_idx = [(4,0), (4,1), (3,1)]
            elif self.n_danger == 4: d_idx = [(4,0), (4,1), (3,1), (3,3)]
            elif self.n_danger == 5: d_idx = [(4,0), (4,1), (3,1), (3,2), (3,3)]
            elif self.n_danger == 6: d_idx = [(1,2), (4,0), (4,1), (3,1), (3,2), (3,3)]
            for d in d_idx: label[d] = ('d',)
            # Colors of the labels 
            lcmap={
                ('a',):'yellow',
                ('b',):'greenyellow',
                ('c',):'turquoise',
                ('d',):'pink'
            }

            grid_mdp = GridMDP(shape=shape,structure=structure,label=label,lcmap=lcmap, p=p, figsize=5)  # Use figsize=4 for smaller figures
        
        elif name == "new_case":
            # MDP Description
            shape = (10,20)
            # E: Empty, T: Trap, B: Obstacle
            structure = np.array([
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E']
            ])

            # Labels of the states
            label = np.array([
            [(),    (),    (),    (),('c',),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),('b',),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),('d',),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),('a',),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()]
            ],dtype=object)
            # Colors of the labels 
            lcmap={
                ('a',):'yellow',
                ('b',):'greenyellow',
                ('c',):'turquoise',
                ('d',):'pink'
            }

            grid_mdp = GridMDP(shape=shape,structure=structure,label=label,lcmap=lcmap, p=p, figsize=14)  # Use figsize=4 for smaller figures
        
        elif name == "frozen_lake":
            # MDP Description
            shape = (15,20)
            # E: Empty, T: Trap, B: Obstacle
            structure = np.array([
            ['B',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['B',  'B',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'B',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'B',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'B',  'B',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'B',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'B',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'B', 'B',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'B',  'B', 'B',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E', 'E',  'E',  'B',  'B',  'B', 'B',  'E',  'E',  'E',  'E', 'E',  'E',  'E',  'E',  'E']
            ])

            # Labels of the states
            label = np.array([
            [(),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('d',),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('d',),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('d',),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('i',),('d',),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),('i',),('i',),('i',),('i',),('i',),('i',),('i',),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),('i',),('i',),('i',),('i',),('i',),('i',),    (),    (),    (),    (),    (),    (),    (),    (),    (),    ()],
            [(),('k',),    (),    (),    (),('i',),('i',),('i',),('i',),    (),    (),('d',),('d',),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),('i',),('i',),('i',),    (),    (),('d',),('d',),    (),    (),    (),('c',),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),('i',),('i',),    (),    (),('d',),('d',),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),('i',),    (),    (),    (),('d',),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),('d',),    (),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),('i',),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),('i',),('i',),    (),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),('i',),('i',),('i',),('i',),    (),    (),    (),    (),    ()],
            [(),    (),    (),    (),    (),    (),    (),    (),    (),    (),    (),('i',),('i',),('i',),('i',),    (),    (),    (),    (),    ()]
            ],dtype=object)
            # Colors of the labels 
            lcmap={
                ('i',):'cyan',
                ('k',):'yellow',
                ('c',):'turquoise',
                ('d',):'pink'
            }

            grid_mdp = GridMDP(shape=shape,structure=structure,label=label,lcmap=lcmap, p=p, figsize=14)  # Use figsize=4 for smaller figures

        return grid_mdp
    
    def get_ltl(self, name):
        if name == 'random':
            ltl = ("(G !d) & ((!c) U b) & ((!b) U a) & (F G c)")
        if name == "sequential_delivery":
            ltl = ("(G !d) & ((!c) U b) & ((!b) U a) & (F G c)")
        elif name == "new_case":
            ltl = ("(G !d) & ((!c) U b) & ((!b) U a) & (F G c)")
        elif name == "frozen_lake":
            ltl = ("(G !d)")
        return ltl

    def set_ltl_f(self):
        if self.name == "random":
            ltl = "[] ~d /\ (~c % b) /\ (~b % a) /\ (<> [] c)"
        if self.name == "sequential_delivery":
            ltl = "[] ~d /\ (~c % b) /\ (~b % a) /\ (<> [] c)"
        elif self.name == "new_case":
            ltl = "[] ~d /\ (~c % b) /\ (~b % a) /\ (<> [] c)"
        elif self.name == "frozen_lake":
            ltl = "[] ~d"

        self.LTL_formula = parser.parse(ltl)
        self.predicates=get_predicates(self.mdp)

    def generate_gw(self, shape):
        
        self.shape = shape

        # E: Empty, T: Trap, B: Obstacle
        self.structure = np.array([['E' for i in range(shape[1])] for i in range(shape[0])])
        # Labels of the states
        self.label = [[() for i in range(shape[1])] for i in range(shape[0])]
        
        self.add_item(1, ('a',))
        self.add_item(1, ('b',))
        self.add_item(1, ('c',))
        self.add_item((shape[0]+shape[1])//2, ('d',)) # danger zones
        self.label = np.array(self.label, dtype=object)
        
        # self.add_obstacles(max_danger_zones = (shape[0]+shape[1])//2)
        # Colors of the labels 
        lcmap={
            ('a',):'yellow',
            ('b',):'greenyellow',
            ('c',):'turquoise',
            ('d',):'pink'
        }

        self.mdp = GridMDP(shape=shape,structure=self.structure,label=self.label,lcmap=lcmap, p=self.p, figsize=(shape[0]+shape[1])//2)  # Use figsize=4 for smaller figures
        self.ltl = self.get_ltl(self.name)
        self.csrl = self.build(self.mdp, self.ltl, plot=self.plot)
        self.set_ltl_f()
    
    def add_item(self, n, item, trap=False):
        for _ in range(n):
            i,j = np.random.randint(self.shape[0]), np.random.randint(self.shape[1])
            while(self.label[i][j] != () or self.structure[i,j]!='E'):
                i,j = np.random.randint(self.shape[0]), np.random.randint(self.shape[1])
            self.label[i][j] = item
            if trap: self.structure[i,j] = 'T'

    def add_obstacles(self, max_obstacles):
        # E: Empty, T: Trap, B: Obstacle
        obstacles_in_row = self.shape[0]//2
        obstacles = 0
        k = 0
        for j in range(1,self.shape[1],2):
            for k in range(obstacles_in_row):
                i = np.random.randint(self.shape[0])
                if self.label[i,j] == () and self.structure[i,j]=='E':
                    self.structure[i,j] = 'B'
                    obstacles += 1
                    if obstacles >= max_obstacles: break
        self.obstacles = obstacles
        