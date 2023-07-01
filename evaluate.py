import lbforaging
import gym
from gym.envs.registration import register
import numpy as np
import time
import argparse
from tqdm import tqdm
import copy
from multiprocessing import Process

from dt import EpsGreedyLeaf, PythonDT, RandomlyInitializedEpsGreedyLeaf
from grammatical_evolution import GrammaticalEvolutionTranslator

import json
import argparse
import skimage.measure
import math

import random

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
parser.add_argument("--genome", default="last_gen_hof.json", type=str, help="JSON file containing a dictionary of genomes.")
parser.add_argument("--players", default=5, type=int, help="Number of players involved.")
parser.add_argument("--field_size", default=15, type=int, help="# of tiles of space lenght. The final game space is a NxN area.")
parser.add_argument("--sight", default=5, type=int, help="# of tiles of view range. The final view is a KxK area.")
parser.add_argument("--sight_downsampling", default=None, type=int, help="Sight downsampling (pooling) factor")
parser.add_argument("--pool_operator", default="mean", type=str, help="Pooling operator: mean, min")
parser.add_argument("--food", default=1, type=int, help="Max # of food laid on the ground.")
parser.add_argument("--cooperation", default=False, type=bool, help="Force players cooperation.")
parser.add_argument("--grid", default=True, type=bool, help="Use the grid observation space.")
parser.add_argument("--max_player_level", default=5, type=int, help="Max achievable player (and food?) level")
parser.add_argument("--multitree", default=False, type=bool, help="Instantiate one DT per agent.")
parser.add_argument("--constant_range", default=1000, type=int, help="Max magnitude for the constants being used (multiplied *10^-3). Default: 1000 => constants in [-1, 1]")
parser.add_argument("--constant_step", default=1, type=int, help="Step used to generate the range of constants, mutliplied *10^-3")
parser.add_argument("--with_bias", action="store_true", help="if used, then the the condition will be (sum ...) < <const>, otherwise (sum ...) < 0")

parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--n_actions", default=4, type=int, help="The number of action that the agent can perform in the environment")
parser.add_argument("--learning_rate", default="auto", help="The learning rate to be used for Q-learning. Default is: 'auto' (1/k)")
parser.add_argument("--df", default=0.9, type=float, help="The discount factor used for Q-learning")
parser.add_argument("--eps", default=0.05, type=float, help="Epsilon parameter for the epsilon greedy Q-learning")
parser.add_argument("--episodes", default=50, type=int, help="Number of episodes that the agent faces in the fitness evaluation phase")
parser.add_argument("--episode_len", default=1000, type=int, help="The max length of an episode in timesteps")

parser.add_argument("--low", default=-10, type=float, help="Lower bound for the random initialization of the leaves")
parser.add_argument("--up", default=10, type=float, help="Upper bound for the random initialization of the leaves")

args = parser.parse_args()

if args.pool_operator == "mean": 
    pool_operator = np.mean
elif args.pool_operator == "max": 
    pool_operator = np.max

DEFAULT_EPSILON = 0.3

# Load from JSON if possible
if args.config_file:
    with open(args.config_file, "r") as agsCfgFile:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(agsCfgFile))
        args = parser.parse_args(namespace=t_args)

np.random.seed(args.seed)

# ------------------------------------------------------------------
#                   SETUP
# ------------------------------------------------------------------

S_ACTIONS = ["NONE", "NORTH", "SOUTH", "WEST", "EAST", "LOAD"]
MAX_STEPS = 1000
N_EPISODES = 50
refresh_time = 0.05

lr = "auto" if args.learning_rate == "auto" else float(args.learning_rate)

# ------------------------------------------------------------------
#                   CLASSES & FUNCTIONS
# ------------------------------------------------------------------

# Creation of an ad-hoc Leaf class
class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    def __init__(self):
        super(CLeaf, self).__init__(args.n_actions, lr, args.df, DEFAULT_EPSILON, low=args.low, up=args.up)

class ListWithParents(list):
    """
        List with some parents attribute 
    """
    def __init__(self, *iterable):
        super(ListWithParents, self).__init__(*iterable)
        self.parents = []

# ------------------------------------------------------------------
#                   DEFINING THE GRAMMAR
# ------------------------------------------------------------------


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


# Setup of the grammar
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

grammar = get_oblique_split_grammar(input_types)

print(f"Grammar len:\t{len(grammar.keys())}")

# ------------------------------------------------------------------
#                   GYM ENVIRONMENT
# ------------------------------------------------------------------

# Load or firstly register the env with the given settings
env_id = "Foraging-{0}x{0}-{1}p-{2}f{3}-v0".format(args.sight, args.players, args.field_size, "-coop" if args.cooperation else "")

print(f"Starting environment:\t {env_id}")

# ------------------------------------------------------------------
#                   GYM ENV PLAYING
# ------------------------------------------------------------------

def visualize_one_run(genome):
    """
        Each processor runs an episode
    """

    cum_episode_rewards = np.zeros(args.players)

    for iteration in range(args.episodes):

        # ---- Switch players at every episode
        if args.multitree:

            population = [x for x in genome.values()]

            # Random population sample
            if args.players < len(population):
                selected_sample = random.choices(list(range(len(population))), k = args.players)
                print(f"Selected sample:\t{selected_sample}")
                selected_sample = np.array(population)[selected_sample]
            else: 
                selected_sample = population

            agents = []
            for ind in selected_sample:
                phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(ind)
                agent = PythonDT(phenotype, CLeaf) # object type
                agents.append(agent)
        else:
            # Convert agent genome => phenotype => DT
            phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(genome)
            agent = PythonDT(phenotype, CLeaf) # object type

        env.seed(iteration)
        observations = env.reset()
        
        episode_rewards = np.zeros(args.players)
        
        # Prepare
        if args.multitree: 
            for agent in agents: agent.new_episode()
        else:
            agent.new_episode()

        # Fixed step to evaluate agents
        for t in range(args.episode_len):

            # Each agent acts
            actions = []
            for i in range(args.players):

                # Downsampling if applies
                if args.sight_downsampling:
                    obs_i = np.stack(
                        [skimage.measure.block_reduce(channel, (args.sight_downsampling, args.sight_downsampling), np.max) for channel in observations[i][:2]]
                    )
                else: 
                    obs_i = observations[i][:2]

                # if i == 0: print(obs_i)

                obs_i = obs_i.reshape(2 * GRID_SIZE * GRID_SIZE) 
                
                if args.multitree:
                    action = agents[i](obs_i)
                else:
                    action = agent(obs_i)

                if action is None: action = 0
                
                #action = action if action is not None else 0
                actions.append(action)

            # Update game state
            observations, curr_rewards, done, info = env.step(actions)

            curr_rewards -= np.ones_like(curr_rewards) * (args.time_penalty)

            #print(actions, curr_rewards)
            #print("-------------")

            if args.multitree:
                for i, r in enumerate(curr_rewards): agents[i].set_reward(r)
            else:
                agent.set_reward(np.sum(curr_rewards))

            episode_rewards += np.array(curr_rewards)
                
            # If all done: early stop
            if not (False in done): break

            # Render last episode's step
            env.render()
            time.sleep(refresh_time)

        cum_episode_rewards += episode_rewards
        #print(f"Episode rewards: \t {episode_rewards}")
        print(f"Cum. episode rewards: \t {cum_episode_rewards}")

# ------------------------------------------------------------------
#                   Script exec
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    try:
        env = gym.make(env_id)
    except:
        register(
            id=env_id,
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": args.players,
                "max_player_level": 3,
                "field_size": (args.field_size, args.field_size),
                "max_food": args.food,
                "sight": args.sight,
                "max_episode_steps": args.episode_len,
                "force_coop": args.cooperation,
                "grid_observation": args.grid
            },
        )    
        env = gym.make(env_id)
 

    with open(args.genome) as json_file:

        # Read the first individual's genome
        genome = json.load(json_file)
        visualize_one_run(genome)

    env.close()
