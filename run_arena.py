import os
import gym
import json
import string
import datetime
import argparse
import lbforaging
import subprocess
import numpy as np
from tqdm import tqdm
from numpy import random
from time import time, sleep
from gym.envs.registration import register
import math

from dt import EpsGreedyLeaf, PythonDT, RandomlyInitializedEpsGreedyLeaf
from grammatical_evolution import GrammaticalEvolutionTranslator, grammatical_evolution, differential_evolution
import skimage.measure

# run this again with the params from the paper


# ------------------------------------------------------------------
#                   CLASSES & FUNCTIONS
# ------------------------------------------------------------------

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

class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    """
        A basic EpsGreedy Leaf instance.
    """
    def __init__(self):
        super(CLeaf, self).__init__(args.n_actions, lr, args.df, args.eps, low=args.low, up=args.up)


class EpsilonDecayLeaf(RandomlyInitializedEpsGreedyLeaf):
    """
        A eps-greedy leaf with epsilon decay.
    """
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

# ----------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--config_file", default=None, type=str, help="(optional) path to a .json args file. Missing params will be applied by default")
parser.add_argument("--field_size", default=15, type=int, help="# of tiles of space lenght. The final game space is a NxN area.")
parser.add_argument("--sight", default=1, type=int, help="# of tiles of view range. The final view is a KxK area.")
parser.add_argument("--sight_downsampling", default=None, type=int, help="Sight downsampling (pooling) factor")
parser.add_argument("--pool_operator", default="mean", type=str, help="Pooling operator: mean, min")
parser.add_argument("--food", default=1, type=int, help="Max # of food laid on the ground.")
parser.add_argument("--cooperation", default=False, type=bool, help="Force players cooperation.")
parser.add_argument("--grid", default=True, type=bool, help="Use the grid observation space.")
parser.add_argument("--max_player_level", default=5, type=int, help="Max achievable player (and food?) level")
parser.add_argument("--encouragement", default=1, type=int, help="Reward multiplier for collaborative collects")
parser.add_argument("--time_penalty", default=0, type=int, help="Subtract some reward score along timesteps")
parser.add_argument("--normalize_reward", default=False, type=bool, help="Normalize rewards by players and food no.")

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
parser.add_argument("--epsDecay", default=False, type=bool, help="Enable decaying eps leaf")
parser.add_argument("--constant_range", default=1000, type=int, help="Max magnitude for the constants being used (multiplied *10^-3). Default: 1000 => constants in [-1, 1]")
parser.add_argument("--constant_step", default=1, type=int, help="Step used to generate the range of constants, mutliplied *10^-3")
parser.add_argument("--with_bias", action="store_true", help="if used, then the the condition will be (sum ...) < <const>, otherwise (sum ...) < 0")

parser.add_argument("--genotype_len", default=100, type=int, help="Length of the fixed-length genotype")
parser.add_argument("--low", default=-10, type=float, help="Lower bound for the random initialization of the leaves")
parser.add_argument("--up", default=10, type=float, help="Upper bound for the random initialization of the leaves")

parser.add_argument("--dt_type", default="orthogonal", required=True, choices=["orthogonal", "oblique"], type=str, help="The type of splits in the DT nodes (orthogonal/oblique)")


# Parse args
args = parser.parse_args()

# Load from JSON if possible
if args.config_file:
    print(f"[-] Running experiment: {args.config_file}...")
    with open(args.config_file, "r") as agsCfgFile:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(agsCfgFile))
        args = parser.parse_args(namespace=t_args)

best = None
lr = "auto" if args.learning_rate == "auto" else float(args.learning_rate)

if args.pool_operator == "mean": 
    pool_operator = np.mean
elif args.pool_operator == "max": 
    pool_operator = np.max

# ------------------------------------------------------------------
#                   SETUP
# ------------------------------------------------------------------

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

if args.epsDecay:
    leaf_class = EpsilonDecayLeaf
else:
    leaf_class = EpsilonDecayLeaf
    
# ------------------------------------------------------------------
#                   DEFINING THE GRAMMAR
# ------------------------------------------------------------------

def get_orthogonal_split_grammar(input_types):
    grammar = {
        "bt": ["<if>"],
        "if": ["if <condition>:{<action>}else:{<action>}"],
        "condition": ["_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(N_INPUT_VARIABLES)],
        "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
        "comp_op": [" < ", " > "],
    }
    for index, input_var in enumerate(input_types):
        start, stop, step, divisor = map(int, input_var)
        consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)])) # np.linspace?
        grammar["const_type_{}".format(index)] = consts_ # add to the grammar values as symbols const_type_x => 0, 0.1, 0.2 (...)
    
    return grammar


def get_oblique_split_grammar(input_types):
    consts = {}

    for index, input_var in enumerate(input_types):
        start, stop, step, divisor = map(int, input_var)

        assert divisor != 0, "Invalid divisor (division by zero)"
        
        consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)])) # np.linspace?
        consts[index] = (consts_[0], consts_[-1])

    oblique_split = "+".join(["<const> * (_in_{0} - {1})/({2} - {1})".format(i, consts[i][0], consts[i][1]) for i in range(input_space_size)])

    grammar = {
        "bt": ["<if>"],
        "if": ["if <condition>:{<action>}else:{<action>}"],
        "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
        # "const": ["0", "<nz_const>"],
        # there is an issue here with some zero divide
        "const": [str(k/1000) for k in range(-args.constant_range,args.constant_range+1,args.constant_step)]
    }

    if not args.with_bias:
        grammar["condition"] = [oblique_split + " < 0"]
    else:
        grammar["condition"] = [oblique_split + " < <const>"]    
        
    return grammar

# ---- Setup of the grammar
if args.sight_downsampling: 
    GRID_SIZE = math.ceil((2 + args.sight * 2) / args.sight_downsampling)
else: 
    GRID_SIZE = (1 + args.sight * 2)

N_INPUT_VARIABLES = 2 * GRID_SIZE * GRID_SIZE # for each player

input_space_size = N_INPUT_VARIABLES

DIVISOR = 1
STEP = 1
FOOD_MIN = 0
FOOD_MAX = args.max_player_level * DIVISOR
AGENT_MIN = 0
AGENT_MAX = args.max_player_level * DIVISOR
ACCESS_MIN = 0
ACCESS_MAX = 1 * DIVISOR
input_types = [AGENT_MIN, AGENT_MAX, STEP, DIVISOR] * (GRID_SIZE * GRID_SIZE) + \
    [FOOD_MIN, FOOD_MAX, STEP, DIVISOR] * (GRID_SIZE * GRID_SIZE)

input_types = np.array(input_types).reshape(2 * GRID_SIZE * GRID_SIZE, 4)

# Grammar depends on the chosen type of split
if args.dt_type == "oblique":
    grammar = get_oblique_split_grammar(input_types)

elif args.dt_type == "orthogonal":
    grammar = get_orthogonal_split_grammar(input_types)

# ------------------------------------------------------------------
#                   GYM ENVIRONMENT
# ------------------------------------------------------------------

# Seeding of the random number generators
random.seed(args.seed)
np.random.seed(args.seed)

# Log all the parameters
with open(logfile, "a") as f:
    vars_ = locals().copy()
    for k, v in vars_.items():
        f.write("{}: {}\n".format(k, v))

def evaluate_fitness(fitness_function, leaf, genes, episodes=args.episodes):
    """
        DT instantiation and evaluation
        One gene, multiple instances (for accurate reward assignment)
    """
    agents = []
    for gene in genes:
        phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(gene["individual"])
        bt = PythonDT(phenotype, leaf) # object type
        agents.append(bt)

    """
    for i in range(args.players):
        gene = genes[i % len(genes)] # rotating gene across players
        phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(gene["individual"])
        bt = PythonDT(phenotype, leaf) # object type
        agents.append(bt)
    """

    return fitness_function(agents, gen=genes[0]["gen"], episode=genes[0]["episode"])


def episode_parallel_fitness(agents, gen=None, episode=None):
    """
        Gym Environment Fitness Evaluation
    """

    # ---- Environment
    env_id = "Foraging-{0}x{0}-{1}p-{2}f{3}-v0".format(args.sight, args.lambda_, args.field_size, "-coop" if args.cooperation else "")

    try:
        e = gym.make(env_id)
    except:
        register(
            id=env_id,
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": args.lambda_,
                "max_player_level": args.max_player_level,
                "field_size": (args.field_size, args.field_size),
                "max_food": args.food,
                "sight": args.sight,
                "max_episode_steps": args.episode_len,
                "force_coop": args.cooperation,
                "grid_observation": args.grid,
                "collab_encouragement": args.encouragement,
                "normalize_reward": args.normalize_reward
            },
        )
        e = gym.make(env_id)

    # ---- Init
    e.seed(episode)
    obs = e.reset()

    for agent in agents: agent.new_episode()
    
    ep_cumulative_rewards = np.zeros(args.lambda_)

    # ---- Episode loop
    for t in range(args.episode_len):

        # ---- Acting
        actions = []
        for i, agent in enumerate(agents):
            
            # Downsampling if applies
            if args.sight_downsampling:
                obs_i = np.stack(
                    [skimage.measure.block_reduce(channel, (args.sight_downsampling, args.sight_downsampling), pool_operator) for channel in obs[i][:2]]
                )
            else: 
                obs_i = obs[i][:2]

            observation = obs_i.reshape(2 * GRID_SIZE * GRID_SIZE) 

            action = agent(observation) # same agent impersonates all players
            action = action if action is not None else 0
            actions.append(action)

        # ---- Environment response
        obs, rewards, done, info = e.step(actions)

        # ---- Rewards
        rewards -= np.ones_like(rewards) * (args.time_penalty)

        for i, r in enumerate(rewards): agents[i].set_reward(r)

        ep_cumulative_rewards += rewards

        if not (False in done): break

    # ---- Log history
    if gen is not None:
        try:
            if not os.path.exists(f"{statsdir}/generations/gen_{gen}"):
                os.makedirs(f"{statsdir}/generations/gen_{gen}")

            for i, p in enumerate(e.players):
                with open(f"{statsdir}/generations/gen_{gen}/player_{i}_stats.json", 'a') as outfile:
                    outfile.write(f"{json.dumps({'episode': episode, 'reward': ep_cumulative_rewards[i], 'actions': p.stat_load_history})}\n")
        except:
            print(f"[!] Error in storing stats for player {i}, gen {gen}")

    e.close()

    # Relative group fitness (social contribution)
    # If players have not contributed much to the group fitness => low
    # ep_cumulative_rewards /= ep_cumulative_rewards.sum()

    return ep_cumulative_rewards


def fit_fcn(Xs):
    return evaluate_fitness(fitness_function=episode_parallel_fitness, leaf=leaf_class, genes=Xs)

# ------------------------------------------------------------------
#                   Script exec
# ------------------------------------------------------------------

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
        pop, log, hof, best_leaves = grammatical_evolution(fit_fcn, episodes=args.episodes, multi_genes=True, statsdir=statsdir, inputs=input_space_size, leaf=leaf_class, individuals=args.lambda_, generations=args.generations, jobs=args.jobs, cx_prob=args.cxp, m_prob=args.mp, logfile=logfile, seed=args.seed, mutation=args.mutation, crossover=args.crossover, initial_len=args.genotype_len, selection=args.selection)

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

