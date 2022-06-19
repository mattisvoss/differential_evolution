#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:16:35 2021

@author: mattis
"""

import numpy as np
import random

# Objective function to optimise
def objective(x):
    return x**2

# Bounds of search space: tuple representing
# (xmin, xmax) for each each input variable

dimensions = 1
bounds = np.empty([dimensions, 2])
bounds[0] = [-2, 2]
population_size = 2 #default in scipy.optimize
max_iterations = 50

def evolve(objective, bounds, population_size, max_generations):
    """Basic Differential Evolution Algorithm"""    
    
    # Initialise a random population of chromosomes within each of the bounds
    population = np.empty(population_size)
    for i in range(0, population_size):
        chromosome = np.empty(len(bounds))        
        for j in range(len(bounds)):
            chromosome[j] = random.uniform(bounds[j,0], bounds[j,1])
        population[i] = chromosome
    #Main evolutionary loop
    iterations = 0
    while iterations <= max_generations:
        iterations += 1
        stopping_criterion = error <= 0.01 
        if stopping_criterion:
            return best_chromosome
            break


c = evolve(objective, bounds, population_size, max_iterations)



