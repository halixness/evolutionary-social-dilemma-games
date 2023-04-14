# Evolutionary Social Dilemma Games
This is a repo for the capstone project of the "Bio-Inspired AI" at the University of Trento. Agents are represented by decision trees, whose structure is defined by an evolving grammar, that is the genome. We seek for understanding of possible emergent social strategies emphasised by natural selection and mating, while using interpretable algorithms.  

### Supported MARL environments
- [Level-based foraging](https://github.com/semitable/lb-foraging)

### Execution

Parameters:
```
    usage: run_base_script.py [-h] [--players PLAYERS] [--field_size FIELD_SIZE] [--sight SIGHT] [--food FOOD]
                          [--cooperation COOPERATION] [--grid GRID] [--max_player_level MAX_PLAYER_LEVEL] [--jobs JOBS]
                          [--seed SEED] [--n_actions N_ACTIONS] [--learning_rate LEARNING_RATE] [--df DF] [--eps EPS]
                          [--episodes EPISODES] [--episode_len EPISODE_LEN] [--lambda_ LAMBDA_] [--generations GENERATIONS]
                          [--cxp CXP] [--mp MP] [--mutation MUTATION] [--crossover CROSSOVER] [--selection SELECTION]
                          [--genotype_len GENOTYPE_LEN] [--low LOW] [--up UP]

    optional arguments:
    -h, --help            show this help message and exit
    --players PLAYERS     Number of players involved.
    --field_size FIELD_SIZE
                            # of tiles of space lenght. The final game space is a NxN area.
    --sight SIGHT         # of tiles of view range. The final view is a KxK area.
    --food FOOD           Max # of food laid on the ground.
    --cooperation COOPERATION
                            Force players cooperation.
    --grid GRID           Use the grid observation space.
    --max_player_level MAX_PLAYER_LEVEL
                            Max achievable player (and food?) level
    --jobs JOBS           The number of jobs to use for the evolution
    --seed SEED           Random seed
    --n_actions N_ACTIONS
                            The number of action that the agent can perform in the environment
    --learning_rate LEARNING_RATE
                            The learning rate to be used for Q-learning. Default is: 'auto' (1/k)
    --df DF               The discount factor used for Q-learning
    --eps EPS             Epsilon parameter for the epsilon greedy Q-learning
    --episodes EPISODES   Number of episodes that the agent faces in the fitness evaluation phase
    --episode_len EPISODE_LEN
                            The max length of an episode in timesteps
    --lambda_ LAMBDA_     Population size
    --generations GENERATIONS
                            Number of generations
    --cxp CXP             Crossover probability
    --mp MP               Mutation probability
    --mutation MUTATION   Mutation operator. String in the format function-value#function_param_-value_1... The operators from    
                            the DEAP library can be used by setting the function to 'function-tools.<operator_name>'. Default:      
                            Uniform Int Mutation
    --crossover CROSSOVER
                            Crossover operator, see Mutation operator. Default: One point
    --selection SELECTION
                            Selection operator, see Mutation operator. Default: tournament of size 2
    --genotype_len GENOTYPE_LEN
                            Length of the fixed-length genotype
    --low LOW             Lower bound for the random initialization of the leaves
    --up UP               Upper bound for the random initialization of the leaves
```
Example:
```
    python run_base_script.py \
        --jobs 16 --seed 42 --n_actions 6 --learning_rate 0.001 --df 0.05 \
        --episodes 30 --lambda_ 100 \
        --generations 500 --cxp 0.5 --mp 0.5 \
        --low -1 --up 1 \
        --mutation "function-tools.mutUniformInt#low-0#up-40000#indpb-0.05"
```

### Papers
- [Evolutionary Learning of Interpretable Decision Trees | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/10015004)
- [Multi-agent Reinforcement Learning in Sequential Social Dilemmas](http://arxiv.org/abs/1702.03037)
