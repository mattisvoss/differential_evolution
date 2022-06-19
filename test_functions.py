#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:17 2021

@author: mattis
"""
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def ackley(x):
    s1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    s2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(s1) - np.exp(s2) + 20. + np.e

def rosen(x):
    x = np.asarray(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (100 * sum((x1 - x0**2) **2 ) + sum((1 - x0) **2 ))

def sphere(x):
    x = np.asarray(x)
    return sum(x**2)
       

fig = plt.figure(dpi=240)
ax = fig.gca(projection='3d')
X = np.arange(-2, 2, 0.25)
Y = np.arange(-1, 3.25, 0.25)
X, Y = np.meshgrid(X, Y)

surf = ax.plot_surface(X, Y, rosen([X,Y]),cmap=cm.coolwarm)
plt.show()

fig = plt.figure(dpi=240)
ax = fig.gca(projection='3d')
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)

surf = ax.plot_surface(X, Y, ackley([X,Y]),cmap=cm.coolwarm)
plt.show()

fig = plt.figure(dpi=240)
ax = fig.gca(projection='3d')
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)

surf = ax.plot_surface(X, Y, sphere([X,Y]),cmap=cm.coolwarm)
plt.show()