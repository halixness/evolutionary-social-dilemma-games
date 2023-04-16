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

parser.add_argument("--genome", default="last_gen_hof.json", type=str, help="JSON file containing a dictionary of genomes.")
parser.add_argument("--players", default=5, type=int, help="Number of players involved.")
parser.add_argument("--field_size", default=15, type=int, help="# of tiles of space lenght. The final game space is a NxN area.")
parser.add_argument("--sight", default=5, type=int, help="# of tiles of view range. The final view is a KxK area.")
parser.add_argument("--food", default=1, type=int, help="Max # of food laid on the ground.")
parser.add_argument("--cooperation", default=False, type=bool, help="Force players cooperation.")
parser.add_argument("--grid", default=True, type=bool, help="Use the grid observation space.")
parser.add_argument("--max_player_level", default=5, type=int, help="Max achievable player (and food?) level")

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

np.random.seed(args.seed)

# ------------------------------------------------------------------
#                   SETUP
# ------------------------------------------------------------------

S_ACTIONS = ["NONE", "NORTH", "SOUTH", "WEST", "EAST", "LOAD"]
MAX_STEPS = 1000
N_EPISODES = 50
refresh_time = 0.5

lr = "auto" if args.learning_rate == "auto" else float(args.learning_rate)

# ------------------------------------------------------------------
#                   CLASSES & FUNCTIONS
# ------------------------------------------------------------------

# Creation of an ad-hoc Leaf class
class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    def __init__(self):
        super(CLeaf, self).__init__(args.n_actions, lr, args.df, args.eps, low=args.low, up=args.up)

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

    # Convert agent genome => phenotype => DT
    phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(genome)
    agent = PythonDT(phenotype, CLeaf) # object type

    for iteration, e in enumerate(range(args.episodes)):

        observations = env.reset()
        episode_rewards = np.zeros(args.players)

        env.seed(iteration)

        # Prepare
        agent.new_episode()

        # Fixed step to evaluate agents
        for t in range(args.episode_len):

            # Each agent acts
            actions = []
            for i in range(args.players):
                obs = observations[i].reshape(3 * GRID_SIZE * GRID_SIZE) 
                action = agent(obs)
                action = action if action is not None else 0
                actions.append(action)

            # Update game state
            observations, curr_rewards, done, info = env.step(actions)

            agent.set_reward(np.sum(curr_rewards))
            episode_rewards += np.array(curr_rewards)
                
            # If all done: early stop
            if not (False in done): break

            # Render last episode's step
            env.render()
            time.sleep(refresh_time)

        print(f"Episode rewards: \t {episode_rewards}")

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
        genome = list(json.load(json_file).values())[0]
        visualize_one_run(genome)

    env.close()
