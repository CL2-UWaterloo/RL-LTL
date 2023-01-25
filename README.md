# RL-LTL
Solving an LTL specified problem through Self-Play
__________________________________________________

Soppose we have a grid world and an agent witin that grid world. Suppose lso that we have been given a set of LTL formulas specifiyng the rules and laws
that the agent should abide within this grid world. For example, assume we have the following grid world:

A -> Location of the agent

E -> Empty cell

O -> Obsticales

G -> Location of the goal


| E  | O | G |
| ------------- | ------------- | ------------- |
| E  | E  | E  |
| a  | O  | E  |

Also assume we have the following simple LTL specification: Eventually G, and always not O. (Meanting the agent a should reach G withough crossing over obsticales O)

<[]~O /\ <>G>
