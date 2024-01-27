from dependencies.LTL import *
from dependencies.Utility_funcs import *

from dependencies.csrl.mdp import GridMDP
from dependencies.csrl.oa import OmegaAutomaton
from dependencies.csrl import ControlSynthesis
import numpy as np

class grid_world:
    def __init__(self, name, p=1, plot=True):
        self.name = name
        self.p = p
        self.ltl = self.get_ltl(self.name)
        self.gw = self.get_mdp(self.name, self.p)
        self.csrl = self.build(self.gw, self.ltl, plot=plot)
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

    def get_ltl(self, name):
        if name == "sequential_delivery":
            ltl = ("(G !d) & ((!c) U b) & ((!b) U a) & (F G c)")
        return ltl

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
            [(),    (),    ('a',),    (),    ()],
            [(),    (),    ('d',),(),    ()],
            [(),    (),    (),    (),    ()],
            [('c',),('d',),    ('d',),('d',),()],
            [(),    ('d',),    ('b',),(),    ()]
            ],dtype=object)
            # Colors of the labels 
            lcmap={
                ('a',):'yellow',
                ('b',):'greenyellow',
                ('c',):'turquoise',
                ('d',):'pink'
            }

            grid_mdp = GridMDP(shape=shape,structure=structure,label=label,lcmap=lcmap, p=p, figsize=5)  # Use figsize=4 for smaller figures

        return grid_mdp
    
    def set_ltl_f(self):
        if self.name == "sequential_delivery":
            ltl = "[] ~d /\ (~c % b) /\ (~b % a) /\ (<> [] c)"

        self.LTL_formula = parser.parse(ltl)
        self.predicates=get_predicates(self.gw)