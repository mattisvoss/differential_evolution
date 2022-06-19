#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:16:35 2021

@author: mattis voss
"""

import numpy as np

# Define test functions
def sphere(x):
    x = np.asarray(x)
    return sum(x**2)

def rosen(x):
    x = np.asarray(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (100 * sum((x1 - x0**2) **2 ) + sum((1 - x0) **2 ))

def ackley(x):
    s1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    s2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(s1) - np.exp(s2) + 20. + np.e

# Define bounds and parameters for DE
dimensions = 2
bounds = np.empty([2, dimensions])
bounds[0] = [-100, 100]
bounds[1] = [-100, 100]
population_size = 20
max_generations = 1000
F = 0.75
C = 0.9

def simple_DE(obj, dims, bounds, F, pop_size, max_gens, C):
    """Basic Differential Evolution Algorithm"""    
    # Initialise a random population of genomes within each of the bounds
    parents = np.empty([pop_size, dimensions])
    for p in range(pop_size):
        parents[p] = [np.random.uniform(bounds[0,0], bounds[0,1]),
               np.random.uniform(bounds[1,0], bounds[1,1])]
    
    # Initialise empty arrays for mutants and children
    children = np.zeros(np.shape(parents))
    
    # Initialise iteration counter, cost array, and stopping criterion
    iteration = 0
    cost_diff = np.zeros(pop_size)
    stopping_criterion = 0
    
    while not stopping_criterion:       
        for i in range(pop_size):
            
            # Select 3 unique random genomes in addition to current parent
            sampling_pop = [n for n in range(pop_size) if n != i]
            rand_i = np.random.choice(sampling_pop, 3, replace = False)
            
            # u is base genome, v and w are used to create differential vector
            u = parents[rand_i[0]]
            v = parents[rand_i[1]]
            w = parents[rand_i[2]]
                       
            # Create mutant based on weighted differential genome added to base
            mutant = u + ( F * (v - w))
            
            # Run crossover selection loop
            trial = np.empty(dims)
            # Random choice from dimensions
            j_rand = np.random.choice([n for n in range(dims)])
            for j in range(dims):
                if np.random.rand() <= C or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = parents[i,j]

            # Let trial genome compete with parent genome
            parent_cost = obj(parents[i])
            trial_cost = obj(trial)
            cost_diff[i] = parent_cost - trial_cost
            if parent_cost > trial_cost:
                children[i] = trial
            else: 
                children[i] = parents[i]
                
        # Update iterator, check stopping criterion, update parent population
        iteration += 1
        stopping_criterion = (np.sum(np.abs(cost_diff)) <= 1e-6 or iteration == max_gens)
        parents = children
        
    # Return mean of last generation of mutants
    print(f"Iterations to convergence: {iteration}")
    return [np.mean(children[0]), np.mean(children[1])]

# Run algorithm on test functions
sphere_min2= simple_DE(sphere, dimensions, bounds, F, population_size, max_generations, C)
rosen_min2 = simple_DE(rosen, dimensions, bounds, F, population_size, max_generations, C)
ackley_min2 = simple_DE(ackley, dimensions, bounds, F, population_size, max_generations, C)