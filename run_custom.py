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


# ------------------------------------------------------------------
#                   SETUP
# ------------------------------------------------------------------

S_ACTIONS = ["NONE", "NORTH", "SOUTH", "WEST", "EAST", "LOAD"]
MAX_STEPS = 1000
N_EPISODES = 50

p = 20 # players
field_size = p * 5
s = field_size # sight
f = p // 2 # max food
c = False # force cooperation
grid_representation = True # game-grid or food-levels-tensor format
refresh_time = .5 # seconds

elitism = 5 # number of best parents to keep in the population
genome_size = 100 # from source
population_size = p
crossover_probability = .5
mutation_probability = .5 
n_generations = 1000
plot_every_n_gens = 3

low = -1 # bound for the random initialization of the leaves
up = 1 # bound for the random initialization of the leaves
n_actions = 6
eps = .2 # Epsilon parameter for the epsilon greedy Q-learning
lr = "auto" # Learning rate q-learning
df = .9 # Discount factor q-learning

# ------------------------------------------------------------------
#                   CLASSES & FUNCTIONS
# ------------------------------------------------------------------

class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    """
        Eps-greedy Q-learning Leaf
    """
    def __init__(self):
        super(CLeaf, self).__init__(n_actions, lr, df, eps, low=low, up=up)

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

GRID_SIZE = 1 + s * 2 
N_INPUT_VARIABLES = 3 * GRID_SIZE * GRID_SIZE # for each player

grammar = {
    "bt": ["<if>"],
    "if": ["if <condition>:{<action>}else:{<action>}"],
    "condition": ["_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(N_INPUT_VARIABLES)],
    "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
    "comp_op": [" < ", " > "],
}

DIVISOR = 10
STEP = 5
FOOD_MIN = 0
FOOD_MAX = 5 * DIVISOR
AGENT_MIN = 0
AGENT_MAX = 5 * DIVISOR
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
env_id = "Foraging-{0}x{0}-{1}p-{2}f{3}-v0".format(s, p, f, "-coop" if c else "")

print(f"Starting environment:\t {env_id}")

register(
    id=env_id,
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": p,
        "max_player_level": 3,
        "field_size": (field_size, field_size),
        "max_food": f,
        "sight": s,
        "max_episode_steps": 50,
        "force_coop": c,
        "grid_observation": grid_representation
    },
)
env = gym.make(env_id)

# ------------------------------------------------------------------
#                   GYM ENV PLAYING
# ------------------------------------------------------------------
def evaluate(population):
    """
        Function to evaluate individual's fitness
        1 episode by default
    """
    global_cumulative_rewards = []
        
    # Initialize agents
    pop = copy.deepcopy(population)
    for i in range(len(pop)): 
        # Convert agent genome => phenotype => DT
        phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(population[i])
        pop[i] = PythonDT(phenotype, CLeaf) # object type

    # Start evaluation
    for e in tqdm(range(N_EPISODES)):

        # Prepare
        env.seed(e)
        for p in pop: p.new_episode()

        # Run & store rewards
        rewards = run_episode(pop)
        global_cumulative_rewards.append(rewards)
    
    return np.mean(np.array(global_cumulative_rewards), axis=0)

def run_episode(population):
    """
        Each processor runs an episode
    """
    observations = env.reset()
    rewards = np.zeros(population_size)

    # Fixed step to evaluate agents
    for t in range(MAX_STEPS):

        # Each agent acts
        actions = []
        for i, agent in enumerate(population):
            obs = observations[i].reshape(3 * GRID_SIZE * GRID_SIZE) 
            action = agent(obs)
            action = action if action is not None else 0
            actions.append(action)

        # Update game state
        observations, curr_rewards, done, info = env.step(actions)
        rewards += np.array(curr_rewards)

        for i, p in enumerate(population):
            p.set_reward(curr_rewards[i])
            
        # If all done: early stop
        if not (False in done): break

        # Render last episode's step
        #if t == MAX_STEPS - 2 and episode == N_EPISODES - 2: 
        #    env.render()
        #    time.sleep(refresh_time)
    return rewards

def selection_best(population, fitness, n_offspring):
    """
        Probabilistic selection of best
    """
    weights = np.array(fitness) / np.sum(fitness) 

    # Probabilistic selection of the best
    offspring = []
    for i in range(n_offspring):
        ind = random.choices(population, weights=weights)
        offspring.append(ind[0])

    return offspring

def selection_tournament(population, fitness, n_offspring, k = 3):
    """
        K torunament selection
    """

    pop_w_fitness = list(zip(population, fitness))

    offspring = []
    for i in range(n_offspring):
        # Select k best
        pool = random.choices(pop_w_fitness, k = k)
        # And append the fittest 
        best_idx = np.argmax(np.array([p[1] for p in pool]))
        offspring.append(pool[best_idx][0])

    return offspring

def varAnd(population, toolbox, cxpb, mutpb):
    """
        Mate & mutate technique 1
    """
    # Clone
    offspring = [toolbox.clone(ind) for ind in population]
    
    # Set parents
    for i, o in enumerate(offspring):
        o.parents = [i] 

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            offspring[i-1].parents.append(i)
            offspring[i].parents.append(i - 1)
            #del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            #del offspring[i].fitness.values

    return offspring

def varDE(population, fitness, toolbox, cxpb, mutpb):
    """
        Mate & mutate technique 2
    """
    # Clone from population
    offspring = [toolbox.clone(ind) for ind in population]

    # Set parents
    for i, o in enumerate(offspring):
        
        o.parents = [i] 

    # Sort by fitness (best selection)
    b = sorted(range(len(offspring)), key=lambda x: fitness[x], reverse=True)[0]

    # Apply crossover and mutation on the offspring
    for i in range(len(offspring)):
        
        # Swap one element of offspring with best from parents (elitism)
        j, k, l = np.random.choice([a for a in range(len(offspring)) if a != i], 3, False)
        if random.uniform(0, 1) > 1:
            j = b

        # Crossover
        offspring[i] = toolbox.mate(offspring[j],
                                    offspring[k],
                                    offspring[l],
                                    offspring[i])
        offspring[i].parents.append(i)
        del offspring[i].fitness.values

    return offspring

def ge_mate(ind1, ind2, individual):
    """
        Crossover operator
    """
    offspring = tools.cxOnePoint(ind1, ind2)

    if random.random() < 0.5:
        new_offspring = []
        for idx, ind in enumerate([ind1, ind2]):
            _, used = GrammaticalEvolutionTranslator(grammar).genotype_to_str(ind)
            if used > len(ind):
                used = len(ind)
            new_offspring.append(individual(offspring[idx][:used]))

        offspring = (new_offspring[0], new_offspring[1])

    return offspring

def ge_mutate(ind, attribute):
    """
        Mutation operator
    """
    random_int = random.randint(0, len(ind) - 1)
    assert random_int >= 0

    if random.random() < 0.5:
        ind[random_int] = attribute()
    else:
        ind.extend(np.random.choice(ind, size=random_int))
    return ind,

def visualize_one_run(population):
    """
        Each processor runs an episode
    """
    observations = env.reset()

    # Initialize agents
    pop = copy.deepcopy(population)
    for i in range(len(pop)): 
        # Convert agent genome => phenotype => DT
        phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(population[i])
        pop[i] = PythonDT(phenotype, CLeaf) # object type

    # Prepare
    for p in pop: p.new_episode()

    # Fixed step to evaluate agents
    for t in range(MAX_STEPS):

        # Each agent acts
        actions = []
        for i, agent in enumerate(pop):
            obs = observations[i].reshape(3 * GRID_SIZE * GRID_SIZE) 
            action = agent(obs)
            action = action if action is not None else 0
            actions.append(action)

        # Update game state
        observations, curr_rewards, done, info = env.step(actions)

        for i, p in enumerate(pop):
            p.set_reward(curr_rewards[i])
            
        # If all done: early stop
        if not (False in done): break

        # Render last episode's step
        env.render()
        time.sleep(refresh_time)


# ------------------------------------------------------------------
#                   EVOLUTIONARY ALGORITHM
# ------------------------------------------------------------------

from deap import creator, base, tools, algorithms
import random

# Setting individual type and fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", ListWithParents, typecode='d', fitness=creator.FitnessMax)

# Initializer of values for each individual
toolbox = base.Toolbox()
toolbox.register("attribute", random.random)

_max_value = 40000
toolbox.register("attr_bool", random.randint, 0, _max_value)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, genome_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
"""
    mutation={'function': "ge_mutate", 'attribute': None}
    crossover={'function': "ge_mate", 'individual': None}

    for d in [mutation, crossover]:
            if "attribute" in d:
                d['attribute'] = toolbox.attr_bool
            if "individual" in d:
                d['individual'] = creator.Individual
                
    toolbox.register("mate", eval(crossover['function']), **{k: v for k, v in crossover.items() if k != "function"})
    toolbox.register("mutate", eval(mutation['function']), **{k: v for k, v in mutation.items() if k != "function"})
    toolbox.register("evaluate", evaluate)
    # Elitism
    n_best_parents = sorted(list(zip(pop, fitnesses)), reverse=True, key=lambda x: x[1])
    n_best_parents = [p[0] for p in n_best_parents][:elitism]

    # Select the next generation individuals
    offspring = selection_tournament(pop, fitnesses, len(pop) - elitism)
    offspring = varAnd(offspring, toolbox, crossover_probability, mutation_probability)

    pop = n_best_parents + offspring
"""

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=4000, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=2)

def main():

    pop = toolbox.population(n = population_size)

    # For each generation
    for gen in range(n_generations):

        elite = tools.selBest(pop, k=elitism)

        # Play & Evaluate
        offspring = algorithms.varAnd(pop, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)
        fits = evaluate(offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = [fit]

        # Select
        pop = toolbox.select(pop, k=(len(pop) - elitism)) + elite
       
       # Report every X runs
        if gen % plot_every_n_gens == 0: 
            fitnesses = [ind.fitness.values for ind in offspring]
            print(f"\nrun \t mean fitness \t max fitness")
            print(f"{gen} \t {np.mean(fitnesses):.4f} \t {np.max(fitnesses):.4f}\n")
            print(f"Fitnesses: \t {fitnesses}")
    
    env.close()

    return pop


# ------------------------------------------------------------------
#                   Script exec
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()