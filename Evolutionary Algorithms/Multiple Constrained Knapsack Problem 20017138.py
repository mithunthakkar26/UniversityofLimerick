#!/usr/bin/env python
# coding: utf-8

# In[23]:


import random

NBR_ITEMS = 100
MAX_WEIGHT = 1000
MAX_ITEM = 100


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# Create the item dictionary: item name is an integer, and value is
# a (value, weight) 2-uple.

items = {}

# Create random items and store them in the items' dictionary.
for i in range(NBR_ITEMS):
     items[i] = (random.randint(1, 10), random.randint(1, 100))

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy

import matplotlib.pyplot as plt

INDIVIDUAL_LENGTH = 100
POPULATION_SIZE = 500
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.01 # probability for mutating an individual
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 10 # number of individuals that we are going to maintain in the hall of fame. 

toolbox = base.Toolbox()

# random.seed(RANDOM_SEED)
toolbox.register("MultipleConstrained", random.randint, 0, 2)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.MultipleConstrained, INDIVIDUAL_LENGTH)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

alpha = 2
def Knapsack_Fitness(individual):
    weights=[]
    values=[]
    for i in range(len(individual)):
        values.append(items[i][0]*individual[i])
        weights.append(items[i][1]*individual[i])
    value = sum(values)
    weight = sum(weights)
    if weight >= MAX_WEIGHT:
        return (value-(weight-MAX_WEIGHT)*alpha), weight
    return value,weight

toolbox.register("evaluate", Knapsack_Fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/INDIVIDUAL_LENGTH)

# Genetic Algorithm flow:
def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # perform the Genetic Algorithm flow:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,
                                              stats=stats, halloffame=hof)

    # print Hall of Fame info:
    print("Hall of Fame Individuals = ", *hof.items, sep="\n")
    print("Best Ever Individual = ", hof.items[0])
    value,weight = Knapsack_Fitness(hof.items[0])
    print(f"Weight of the best individual: {weight}")
    print(f"Value of the best individual: {value}")
    # Genetic Algorithm is done - extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    # Genetic Algorithm is done - plot statistics:
    #sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()


# In[ ]:




