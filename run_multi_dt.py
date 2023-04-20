import os
import gym
import json
import string
import datetime
import argparse
import subprocess
import numpy as np
from time import time, sleep
from numpy import random
from dt import EpsGreedyLeaf, PythonDT, RandomlyInitializedEpsGreedyLeaf
from grammatical_evolution import GrammaticalEvolutionTranslator, grammatical_evolution, differential_evolution
import lbforaging
from gym.envs.registration import register


def string_to_dict(x):
    """
    This function splits a string into a dict.
    The string must be in the format: key0-value0#key1-value1#...#keyn-valuen
    """
    result = {}
    items = x.split("#")

    for i in items:
        key, value = i.split("-")
        try:
            result[key] = int(value)
        except:
            try:
                result[key] = float(value)
            except:
                result[key] = value

    return result

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default=None, type=str, help="(optional) path to a .json args file. Missing params will be applied by default")
parser.add_argument("--players", default=5, type=int, help="Number of players involved.")
parser.add_argument("--field_size", default=15, type=int, help="# of tiles of space lenght. The final game space is a NxN area.")
parser.add_argument("--sight", default=5, type=int, help="# of tiles of view range. The final view is a KxK area.")
parser.add_argument("--food", default=1, type=int, help="Max # of food laid on the ground.")
parser.add_argument("--cooperation", default=False, type=bool, help="Force players cooperation.")
parser.add_argument("--grid", default=True, type=bool, help="Use the grid observation space.")
parser.add_argument("--max_player_level", default=5, type=int, help="Max achievable player (and food?) level")

parser.add_argument("--jobs", default=1, type=int, help="The number of jobs to use for the evolution")
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--n_actions", default=4, type=int, help="The number of action that the agent can perform in the environment")
parser.add_argument("--learning_rate", default="auto", help="The learning rate to be used for Q-learning. Default is: 'auto' (1/k)")
parser.add_argument("--df", default=0.9, type=float, help="The discount factor used for Q-learning")
parser.add_argument("--eps", default=0.4, type=float, help="Epsilon parameter for the epsilon greedy Q-learning")
parser.add_argument("--episodes", default=50, type=int, help="Number of episodes that the agent faces in the fitness evaluation phase")
parser.add_argument("--episode_len", default=500, type=int, help="The max length of an episode in timesteps")
parser.add_argument("--lambda_", default=30, type=int, help="Population size")
parser.add_argument("--generations", default=1000, type=int, help="Number of generations")
parser.add_argument("--cxp", default=0.5, type=float, help="Crossover probability")
parser.add_argument("--mp", default=0.5, type=float, help="Mutation probability")
parser.add_argument("--mutation", default="function-tools.mutUniformInt#low-0#up-40000#indpb-0.1", type=string_to_dict, help="Mutation operator. String in the format function-value#function_param_-value_1... The operators from the DEAP library can be used by setting the function to 'function-tools.<operator_name>'. Default: Uniform Int Mutation")
parser.add_argument("--crossover", default="function-tools.cxOnePoint", type=string_to_dict, help="Crossover operator, see Mutation operator. Default: One point")
parser.add_argument("--selection", default="function-tools.selTournament#tournsize-2", type=string_to_dict, help="Selection operator, see Mutation operator. Default: tournament of size 2")
parser.add_argument("--stats", default=None, type=str, help="Specific folder to store the stats for an experiment. Default is yyyy-mm-dd-HH-MM.")
parser.add_argument("--random_init", action="store_true", help="Randomly initializes the leaves in [-1, 1[")
parser.add_argument("--decay", default=0.99, type=float, help="The decay factor for the epsilon decay (eps_t = eps_0 * decay^t)")

parser.add_argument("--genotype_len", default=100, type=int, help="Length of the fixed-length genotype")
parser.add_argument("--low", default=-10, type=float, help="Lower bound for the random initialization of the leaves")
parser.add_argument("--up", default=10, type=float, help="Upper bound for the random initialization of the leaves")

# Parse args
args = parser.parse_args()

# Load from JSON if possible
if args.config_file:
    with open(args.config_file, "r") as agsCfgFile:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(agsCfgFile))
        args = parser.parse_args(namespace=t_args)

best = None
lr = "auto" if args.learning_rate == "auto" else float(args.learning_rate)

# Setup of the logging
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = "logs/gym/{}_{}".format(date, "".join(np.random.choice(list(string.ascii_lowercase), size=8)))
logfile = os.path.join(logdir, "log.txt")
os.makedirs(logdir)

if args.stats:
    statsdirname = args.stats
else:
    statsdirname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

statsdir = "stats/{}".format(statsdirname)
if not os.path.exists(statsdir): 
    os.makedirs(statsdir)
    os.makedirs(f"{statsdir}/generations")

# Creation of an ad-hoc Leaf class
class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    def __init__(self):
        super(CLeaf, self).__init__(args.n_actions, lr, args.df, args.eps, low=args.low, up=args.up)


class EpsilonDecayLeaf(RandomlyInitializedEpsGreedyLeaf):
    """A eps-greedy leaf with epsilon decay."""

    def __init__(self):
        """
        Initializes the leaf
        """
        if not args.random_init:
            RandomlyInitializedEpsGreedyLeaf.__init__(
                self,
                n_actions=args.n_actions,
                learning_rate=lr,
                discount_factor=args.df,
                epsilon=args.eps,
                low=0,
                up=0
            )
        else:
            RandomlyInitializedEpsGreedyLeaf.__init__(
                self,
                n_actions=args.n_actions,
                learning_rate=lr,
                discount_factor=args.df,
                epsilon=args.eps,
                low=-1,
                up=1
            )

        self._decay = args.decay
        self._steps = 0

    def get_action(self):
        self.epsilon = self.epsilon * self._decay
        self._steps += 1
        return super().get_action()


# Setup of the grammar
GRID_SIZE = 1 + args.sight * 2 
N_INPUT_VARIABLES = 3 * GRID_SIZE * GRID_SIZE # for each player

input_space_size = N_INPUT_VARIABLES

grammar = {
    "bt": ["<if>"],
    "if": ["if <condition>:{<action>}else:{<action>}"],
    "condition": ["_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(N_INPUT_VARIABLES)],
    "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
    "comp_op": [" < ", " > "],
}

DIVISOR = 1
STEP = 5
FOOD_MIN = 0
FOOD_MAX = args.max_player_level * DIVISOR
AGENT_MIN = 0
AGENT_MAX = args.max_player_level * DIVISOR
ACCESS_MIN = 0
ACCESS_MAX = 1 * DIVISOR
input_types = [AGENT_MIN, AGENT_MAX, STEP, DIVISOR] * (GRID_SIZE * GRID_SIZE) + \
    [FOOD_MIN, FOOD_MAX, STEP, DIVISOR] * (GRID_SIZE * GRID_SIZE) + \
    [ACCESS_MIN, ACCESS_MAX, STEP, DIVISOR] * (GRID_SIZE * GRID_SIZE)

input_types = np.array(input_types).reshape(3 * GRID_SIZE * GRID_SIZE, 4)

# For all defined input spaces
for index, input_var in enumerate(input_types):
    start, stop, step, divisor = map(int, input_var)
    consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)])) # np.linspace?
    grammar["const_type_{}".format(index)] = consts_ # add to the grammar values as symbols const_type_x => 0, 0.1, 0.2 (...)

# print(f"Grammar len:\t{len(grammar.keys())}")

# Seeding of the random number generators
random.seed(args.seed)
np.random.seed(args.seed)


# Log all the parameters
with open(logfile, "a") as f:
    vars_ = locals().copy()
    for k, v in vars_.items():
        f.write("{}: {}\n".format(k, v))

# Definition of the fitness evaluation function
def evaluate_fitness(fitness_function, leaf, genotype, episodes=args.episodes):
    """
        DT instantiation and evaluation
        One gene, multiple instances (for accurate reward assignment)
    """
    agents = []
    for p in range(args.players):
        # from genotype => phenotype (DT specs) => build DT agent
        phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(genotype["individual"])
        bt = PythonDT(phenotype, leaf) # object type
        agents.append(bt)

    return fitness_function(agents, gen=genotype["gen"], episodes=episodes)


def fitness(agents, gen=None, episodes=args.episodes):
    """
        Gym Environment Fitness Evaluation
    """
    
    # initialize random and rewards
    random.seed(args.seed)
    np.random.seed(args.seed)
    global_cumulative_rewards = np.zeros(args.players)
    episodes_r = []

    # Load or firstly register the env with the given settings
    env_id = "Foraging-{0}x{0}-{1}p-{2}f{3}-v0".format(args.sight, args.players, args.field_size, "-coop" if args.cooperation else "")

    # If the environment does not exist -> registration
    try:
        e = gym.make(env_id)

    except:
        register(
            id=env_id,
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": args.players,
                "max_player_level": args.max_player_level,
                "field_size": (args.field_size, args.field_size),
                "max_food": args.food,
                "sight": args.sight,
                "max_episode_steps": args.episode_len,
                "force_coop": args.cooperation,
                "grid_observation": args.grid
            },
        )
        e = gym.make(env_id)

    # for each episode (initial state)
    for iteration in range(episodes):

        e.seed(iteration)
        
        # start & set leaves to none for current DT
        obs = e.reset()
        for agent in agents: agent.new_episode()
        
        episode_r = 0

        # Cumulative per episode
        #episode_reward = np.zeros(len(agents))

        # Finite horizon
        for t in range(args.episode_len):

            # Each agent acts
            actions = []
            for i, agent in enumerate(agents):
                observation = obs[i].reshape(3 * GRID_SIZE * GRID_SIZE) 
                action = agent(observation) # same agent impersonates all players
                action = action if action is not None else 0
                actions.append(action)

            obs, rewards, done, info = e.step(actions)
            for i, r in enumerate(rewards): agents[i].set_reward(r)

            # Track episode reward and cumulative players reward
            episode_r += np.sum(rewards)
            global_cumulative_rewards += rewards

            # If all done: early stop
            if not (False in done): break

        # End of the episode rewards and exploration (from original code)
        for i, agent in enumerate(agents):
                observation = obs[i].reshape(3 * GRID_SIZE * GRID_SIZE) 
                agent(observation) # same agent impersonates all players
                
        episodes_r.append(episode_r)

    # Store player stats for each generation
    if gen:
        if not os.path.exists(f"{statsdir}/generations/gen_{gen}"):
            os.makedirs(f"{statsdir}/generations/gen_{gen}")

        for i, p in enumerate(e.players):
            with open(f"{statsdir}/generations/gen_{gen}/player_{i}_stats.json", 'a') as outfile:
                    outfile.write(f"{json.dumps(p.stat_load_history)}\n")
            
    # Close and return results
    e.close()
    fitness = np.mean(episodes_r),
    
    # Return avg. cumulative rewards and the best performing agent's leaves?
    return fitness, agents[np.argmax(global_cumulative_rewards)].leaves


def fit_fcn(x):
    return evaluate_fitness(fitness_function=fitness, leaf=EpsilonDecayLeaf, genotype=x)


# Workaround for parallel processing on Windows
# https://stackoverflow.com/questions/61065222/python-deap-and-multiprocessing-on-windows-attributeerror
from deap import base, creator, tools, algorithms
from grammatical_evolution import ListWithParents

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", ListWithParents, typecode='d', fitness=creator.FitnessMax)

if __name__ == '__main__':
    import collections
    from joblib import parallel_backend

    # Storing configuration
    if logdir:
        with open(f"{statsdir}/environment_settings.json", 'w') as outfile:
            json.dump(vars(args), outfile)

    # this only works well on UNIX systems
    with parallel_backend("multiprocessing"):
        pop, log, hof, best_leaves = grammatical_evolution(fit_fcn, statsdir=statsdir, inputs=input_space_size, leaf=EpsilonDecayLeaf, individuals=args.lambda_, generations=args.generations, jobs=args.jobs, cx_prob=args.cxp, m_prob=args.mp, logfile=logfile, seed=args.seed, mutation=args.mutation, crossover=args.crossover, initial_len=args.genotype_len, selection=args.selection)

    # Log best individual
    with open(logfile, "a") as log_:
        phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(hof[0])
        phenotype = phenotype.replace('leaf="_leaf"', '')

        for k in range(50000):  # Iterate over all possible leaves
            key = "leaf_{}".format(k)
            if key in best_leaves:
                v = best_leaves[key].q
                phenotype = phenotype.replace("out=_leaf", "out={}".format(np.argmax(v)), 1)
            else:
                break

        log_.write(str(log) + "\n")
        log_.write(str(hof[0]) + "\n")
        log_.write(phenotype + "\n")
        log_.write("best_fitness: {}".format(hof[0].fitness.values[0]))

    with open(os.path.join(logdir, "fitness.tsv"), "w") as f:
        f.write(str(log))

