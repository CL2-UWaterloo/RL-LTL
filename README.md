# RL-LTL
Solving an LTL specified problem through Self-Play
__________________________________________________

Suppose we have a grid world and an agent within that grid world. Suppose also that we have been given a set of LTL formulas specifying the rules and laws
That the agent should abide within this grid world. For example, assume we have the following grid world:

A -> Location of the agent

E -> Empty cell

O -> Obsticales

G -> Location of the goal


| E  | O | G |
| ------------- | ------------- | ------------- |
| E  | E  | E  |
| a  | O  | E  |

Also assume we have the following simple LTL specification: Eventually G, and always not O. (Meanting the agent a should reach G withough crossing over obstacles O)

<[]~O /\ <>G>

Now, we aim to create policies that abide by this rule and lead to trajectories which satisfy the given specs. We are using an AlphaGo zero approach to this problem, meaning that we have a policy+value network that outputs a policy (for movement) and a value (chances of satisfying the specs) at each time-step. We then run multiple Monte-Carlo rollouts from the current position to improve the given policy, and then, to make a move we sample from the resulting policy.
