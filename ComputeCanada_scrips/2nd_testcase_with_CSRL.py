from NN import *
from LTL import *


from csrl.mdp import GridMDP
from csrl.oa import OmegaAutomaton
from csrl import ControlSynthesis
import numpy as np

### from the Nursary case ###
# LTL Specification
ltl = ('G ('
    '(!d) & '
    '((b & (!(X b)))->(X ((!b) U (a|c)))) & '
    '(((!b) & (X b) & (!(X X b)))->((!a) U c)) & '
    '(a->(X ((!a) U b))) & '
    '(c->((!a) U b)) & '
    '((b & (X b))->(F a))'
')')

# ltl = ('G ((!d) & (c->((!a) U b)) )')

# Translate the LTL formula to an LDBA
oa = OmegaAutomaton(ltl)
print('Number of Omega-automaton states (including the trap state):',oa.shape[1])

# MDP Description
shape = (5,4)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
['E',  'E',  'E',  'E'],
['E',  'E',  'E',  'E'],
['E',  'E',  'E',  'E'],
['E',  'E',  'E',  'E'],
['E',  'E',  'E',  'E']
])

# Labels of the states
label = np.array([
[(),    (),    ('b',),('d',)],
[(),    (),    (),    ()],
[(),    (),    (),    ()],
[('a',),(),    (),    ()],
[(),    ('c',),(),    ()]
],dtype=object)
# Colors of the labels
lcmap={
    ('a',):'yellow',
    ('b',):'greenyellow',
    ('c',):'turquoise',
    ('d',):'pink'
}
grid_mdp = GridMDP(shape=shape,structure=structure,label=label,lcmap=lcmap,p=1,figsize=5)  # Use figsize=4 for smaller figures
grid_mdp.plot()

# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa)

model = build_model((5,4), csrl.shape[-1])

t = "[] ( (~d) /\ (c->(~a % b)) )"
full_t = "[] ( (~d) /\ ((b /\ ~>b) -> >(~b % (a \/ c))) /\ (a -> >(~a % b))"
full_t += " /\ ((~b /\ >b /\ ~>>b)->(~a % c)) /\ (c->(~a % b)) /\ ((b /\ >b) -> <>a))"

LTL_formula = parser.parse(full_t)
predicates={'a':[12], 'b':[2], 'c':[17], 'd':[3]}

### proof of satifying trajectory ###
tra = [17,13,9,5,1,2,6,10,14,18,17]
print(len(tra), check_LTL(LTL_formula, tra, predicates))
#######

trajectory, action_history, reward, better_policy = Q=csrl.MC_learning(model, LTL_formula, predicates, n_steps=10, C=3, tow=1, n_samples=100,
            N=np.zeros(csrl.shape), W=np.zeros(csrl.shape), Q=np.zeros(csrl.shape), P=np.zeros(csrl.shape),
            verbose=0,start=(4,1),T=15,K=100)
###############################################################

print("reward:", reward, "  | trajectory:", trajectory)
print("Actions:", action_history)

