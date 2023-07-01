# Evolution in Social Dilemma Games

## Goal
To study what behaviors emerge in a social game where players evolve.

## Questions
Setting of the plots: for each environment condition, plot the normal and forced coop learning curve.
In each plot add the two diversity curves as well ([plots with multiple scales](https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/two_scales.html)). Add the diversities and also the genetic relatedness curve.

- Does evolution encourage selfish behavior (competition) or cooperation?
See the difference between forced cooperation and spontaneous in average rewards.

- Under what conditions can
See how reward curve changes over multiple conditions
Tested conditions: 
    force coop: 0a, 0b, 0c, 1c, 2c
    normal:     0a, 0b, 0c, 1c, 2c


## Proposed methodology
Evolution of decision tree structures. The leaves are then trained with Q-Learning (to map from a state to an action through classification).
- A genotype is a list of integers
- From a grammar, we can produce nestes if-then-else code by following the production rules
- The genes dictate which rule to take to produce the code according to the grammar (including the constants)
- Input vars are compared with constants (genes) until they reach leaves
- The leaves are then trained with Epsilon-greedy Q-Learning to classify the actions
In summary: a tree is evolve to distinguish the world states (when explored); with q-learning, the leaves (various cases for some states) are trained to pick the best decision for that case.

## Food scarcity
| Level | Condition | Relation (food/person) |
|---|---|---|
| 1 | Scarcity | x < 1 |
| 2 | Neutral | x = 1 |
| 3 | Abundance | x  > 1 |
    
## Social pressure 

From [Proxemics, Edward T. Hall](https://en.wikipedia.org/wiki/Proxemics).

- Personal distance 
    absolute: (0.46m < x <= 1.22m)
    relative: (0.83 < x <= 2.22)

- Social distance 
    absolute: (1.2m < x <= 3.7m)
    relative: (2.22 < x <= 6.73)

- Public distance 
    absolute: (x > 3.7m)
    relative: (x > 6.73)

So the neutral level is **social distance**, which means 2 tiles of distance from each player (a 5x5 square of area). This is **0.04 players/tile** (1 each 25 tiles).

| Level | Condition | Relation (players/tile) |
|---|---|---|
| 1 | Sparse | x <= 0.02 |
| 2 | Neutral | 0. 02 < x <= 0.04 |
| 3 | Threatening | x > 0.04 |

### Relations

Let $S_d$ the subject social distance, $S_s$ the subject size:
$$
\begin{equation}
    1 : S_s = X : S_d
\end{equation}
$$

Let $D_r$ the relative distance, $D_a$ the absolute distance, $S_s$ the subject size:
$$
\begin{equation}
    D_r = \frac{D_a}{S_s}
\end{equation}
$$

Let $d$ the density (persons/tile), $N_p$ the number of players, $T_N \times T_M$ a matrix of tiles:
$$
\begin{equation}
    d = \frac{N_p}{T_N \times T_M}
\end{equation}
$$

### Social pressure in level based foraging

From [Christianos et al. 2019](https://arxiv.org/abs/2006.07169). 

| Condition | Tiles | Players | players/tile | Food | players/food | Level |
|---|---|---|---|---|---|---|
| $a^{[1]}$ | 10x10 | 3 | 0.030 | 3 | 1 | 1 |
| $d^{[1]}$ | 8x8 | 2 | 0.031 | 2 | 1 | 1 |
| $c^{[1]}$ | 15x15 | 3 | 0.013 | 4 | 0.75 | 2.a |
| $b^{[1]}$ | 12x12 | 2 | 0.014 | 1 | 2 | 2.b |

### Social pressure in evolutionary social dilemmas

We identify multiple levels for the above conditions combined:
1. Level 1: neutral, 1 player per food, sparse players.
1. Level 1: neutral, 1 player per food, sparse players. 

Here follow the previous conditions from LBF but scaled to the population considered in this experiment.

| Condition | Tiles | Players | players/tile | Food | players/food | Social pressure | Food scarcity |
|---|---|---|---|---|---|---|---|
| $0.c^{[2]}$ | 62x62 | 50 | 0.013 | 67 | 0.75 | 1 | 3 |
| $0.a^{[2]}$ | 41x41 | 50 | 0.030 | 50 | 1 | 2 | 2 | 
| $0.b^{[2]}$ | 60x60 | 50 | 0.014 | 25 | 2 | 1 | 1 |

New conditions:

| Condition | Tiles | Players | players/tile | Food | players/food | Social pressure | Food scarcity |
|---|---|---|---|---|---|---|---|
| $1.a^{[2]}$ | 35x35 | 50 | 0.04 | 67 | 0.75 | 2 | 3 |
| $1.c^{[2]}$ | 35x35 | 50 | 0.04 | 25 | 2 | 2 | 1 |
| $2.a^{[2]}$ | 25x25 | 50 | 0.08 | 67 | 0.75 | 3 | 3 |
| $2.c^{[2]}$ | 25x25 | 50 | 0.08 | 25 | 2 | 3 | 1 |

### Social utility
From the idea of group selection (Wynne-Edwards, 1986 and Michod, 1999), the population of individuals can be seen as a society. Contributing to the society means doing actions leading to the survival of as many individuals as possible: this ensures diversity and thus new behaviors.
A draft for a "social utility fitness" $F_u$:

$$
\begin{equation}
    \hat{f}(i) = \frac{f(i)}{\sum^{N}_{j} f(j) } 
\end{equation}
$$
$$
\begin{equation}
    F_u(i) = -\sum^{N}_{j} |\hat{f}(i)-\hat{f}(j)| \quad \forall j \neq i
\end{equation}
$$

Firstly, each player's measurement is the relative contribution to the collective fitness. However, this does not prevent a player from maximizing its reward against the others, thus the social utility $F_u$ is defined as the negative total utility gap (to maximize).
This allows to measure each players' reward in comparison to the others, while discouraging surpassing their peers.

### Hamilton's rule

[Kin selection](https://en.wikipedia.org/wiki/Kin_selection) is the phenomenon described by the Hamilton's rule: genetic relatedness $r$ (portion of overlapping genes) is an upper bound for the ratio between the cost of the helper $C$ and the benefit received from the recipient $R$.

$$
\begin{equation}
    r > \frac{C}{R}
\end{equation}
$$
