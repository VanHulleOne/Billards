# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:37:59 2016

@author: lvanhulle
"""

import numpy as np

m, n = 2, 5
pq = np.array([[1,1], [1,4]])

table = np.array([[[0,0], [m,0]], [[m,0],[0,n]], [[0,n], [m,0]], [[0,0],[0,n]]])

collisions = []
travel = np.array([[0,0], [1,1]])

for side in table:
    t = np.cross((travel[0] - side[0]), travel[1])/(1.0*np.cross(side[1], (travel[1])))
    print t

def col_gen():
    for side in table:
        yield np.cross((travel[0] - side[0]), travel[1])/(1.0*np.cross(side[1], (travel[1])))

print any(0 < t and t < 1 for t in col_gen())


