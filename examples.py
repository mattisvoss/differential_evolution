#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:17 2021

@author: mattis
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt

x1 = np.outer(np.linspace(-1, 1, 50), np.ones(50))
x2 = x1.copy().T
def f(x1, x2):
    return( 10 - math.exp(-( x1 + 3 * x2 )))

#ecf = np.vectorize(f)
#y = vecf(x1, x2)

y = 10 - np.exp(-( x1 + 3 * x2 ))

fig = plt.figure(figsize = (14,9))
ax = plt.axes(projection = '3d')
ax.plot_surface(x1, x2, y)
plt.show