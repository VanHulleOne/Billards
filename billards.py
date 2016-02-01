# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:37:59 2016

@author: lvanhulle
"""

import numpy as np
import time

m, n = 2, 3
pq = np.array([[1,1], [0,3]])
POINT, VECT = 0, 1
X, Y = 0,1

#lines are of the form startPoint, line vector
table = np.array([[[0,0], [m,0]], [[m,0],[0,n]], [[m,n], [-m,0]], [[0,n],[0,-n]]])

collisions = [np.array([0.,0.])]
travel = np.array([[0,0], [1,1]])


def getLineConst(line, cLine):
    return (np.cross((line[POINT] - cLine[POINT]), line[VECT])/
        (1.0*np.cross(cLine[VECT], (line[VECT]))))
        
def getTU(uLine, tLine):
    return getLineConst(uLine, tLine), getLineConst(tLine, uLine)

intersection = np.array([0,0])

epsilon = 0.001


for j in xrange(m+n-1):
    for i in xrange(len(table)):
        t, u = getTU(travel, table[i])
        if t >= 0 and t <= 1 and u > 0:
            intersection = table[i][POINT] + t * table[i][VECT]
            collisions.append(intersection)
            travel[0] = intersection
            if i%2:
                travel[VECT][X] *= -1
            else:
                travel[VECT][Y] *= -1

assert(len(collisions) == m+n)
assert(any(np.all(np.equal(line[POINT], (collisions[-1]))) for line in table))          
for c in collisions:
    print c