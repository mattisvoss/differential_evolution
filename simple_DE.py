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

def simple_DE(obj, dims, bounds, F, pop_size, max_gens):
    """Basic Differential Evolution Algorithm"""    
    # Initialise a random population of genomes within each of the bounds
    parents = np.empty([pop_size, dimensions])
    for p in range(pop_size):
        parents[p] = [np.random.uniform(bounds[0,0], bounds[0,1]),
               np.random.uniform(bounds[1,0], bounds[1,1])]
    
    # Initialise empty array for next generation
    children = np.zeros(np.shape(parents))
    
    # Initialise iteration counter, cost array, and stopping criterion
    iteration = 0
    cost_diff = np.zeros(pop_size)
    stopping_criterion = 0
    
    # Add storage for animation of convergence
    storage = np.zeros(1)
    while not stopping_criterion:
        
        # This section is used to save a series of snapshots of algorithm converging
        # Make contour plot of objective function
        #fig,ax=plt.subplots(figsize=(8, 8))
        #plt.figure(figsize=(5,5))
        #ax.contour(X, Y, ackley([X,Y]))
        
        for i in range(pop_size):
            # Selecting 3 unique random genomes in addition to current parent
            sampling_pop = [n for n in range(pop_size) if n != i]
            rand_i = np.random.choice(sampling_pop, 3, replace = False)
            # u is base genome, v and w are used to create difference genome
            u = parents[rand_i[0]]
            v = parents[rand_i[1]]
            w = parents[rand_i[2]]
            # Create mutant based on weighted difference genome added to base
            mutant = u + ( F * (v - w))
            
            # Let mutant genome compete with parent genome
            parent_cost = obj(parents[i])
            mutant_cost = obj(mutant)
            cost_diff[i] = parent_cost - mutant_cost
            if parent_cost > mutant_cost:
                children[i] = mutant
            else: 
                children[i] = parents[i]
        
        # Update iterator, check stopping criterion, update parent population
        iteration += 1
        stopping_criterion = (np.sum(np.abs(cost_diff)) <= 1e-6 or iteration == max_gens)
        parents = children
        
        # Make scatter plot of genomes in each generation
        #print(children)
        #ax.scatter(children[:, 0], children[:, 1], c = 'black')
        #plt.savefig(f'ackley{iteration:00}.pdf')    
    print(f"Iterations to convergence: {iteration}")

    return [np.mean(children[0]), np.mean(children[1])]

# Mesh grid for contour plot of test function
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)

# Run algorithm on test functions
sphere_min2= simple_DE(sphere, dimensions, bounds, F, population_size, max_generations, C)
rosen_min2 = simple_DE(rosen, dimensions, bounds, F, population_size, max_generations, C)
ackley_min2 = simple_DE(ackley, dimensions, bounds, F, population_size, max_generations, C)