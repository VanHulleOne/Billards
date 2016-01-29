# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:37:59 2016

@author: lvanhulle
"""

import numpy as np
import time

m, n = 3, 4
pq = np.array([[1,1], [0,3]])

#lines are of the form startPoint, line vector
table = np.array([[[0,0], [m,0]], [[m,0],[0,n]], [[0,n], [m,0]], [[0,0],[0,n]]])

collisions = []
travel = np.array([[0,0], [1,1]])

for side in table:
    t = np.cross((travel[0] - side[0]), travel[1])/(1.0*np.cross(side[1], (travel[1])))
    print t

#returns the scalar for the intersection on line2
def crosser(line1, line2):
    return np.cross((line1[0] - line2[0]), line1[1])/(1.0*np.cross(line2[1], (line1[1])))
    
def table_cross_gen():
    for side in table:
        yield crosser(travel, side)

intersection = np.array([0,0])
goal = np.array([m,n])

epsilon = 0.001


for j in range(m+n-1):
    for i in xrange(len(table)):
        t = crosser(travel, table[i])
        u = crosser(table[i], travel)
        if t >= 0 and t <= 1 and u > 0:
            intersection = table[i][0] + t * table[i][1]
            print 'Intersection'            
            print intersection
            travel[0] = intersection
            if i%2:
                travel[1][0] *= -1
            else:
                travel[1][1] *= -1
            print 'Travel'
            print travel
            time.sleep(.1)