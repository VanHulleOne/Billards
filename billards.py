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
numCrosses = 0
EPSILON = 0.001

#lines are of the form startPoint, vector from start to end point
table = np.array([[[0,0], [m,0]], [[m,0],[0,n]], [[m,n], [-m,0]], [[0,n],[0,-n]]])

collisions = [np.array([0.,0.])]
travel = np.array([[0,0], [1,1]])

def hasArea(points):
    if len(points) != 3: raise Exception('getArea(points) Illegal number of points. \
                                Must use exactly 3 points.')
    matrix = np.ones([3,3])
    for i in xrange(len(points)):
        matrix[i][0:2] = points[i]
        
    return abs(np.linalg.det(matrix)) > EPSILON
    
def areColinear(line1, line2):
    p2 = line1[POINT] + line1[VECT]
    p4 = line2[POINT] + line2[VECT]
    return not (hasArea((line1[POINT], p2, line2[POINT])) or
            hasArea((line1[POINT], p2, p4)))

def getLineConst(line, cLine):
    return (np.cross((line[POINT] - cLine[POINT]), line[VECT])/
            (1.0*np.cross(cLine[VECT], (line[VECT]))))
        
def getTU(uLine, tLine):
    return getLineConst(uLine, tLine), getLineConst(tLine, uLine)

intersection = np.array([0,0])




for j in xrange(m+n-1):
    for side in xrange(len(table)):
        t, u = getTU(travel, table[side])
        if t >= 0 and t <= 1 and u > 0:
            intersection = table[side][POINT] + t * table[side][VECT]
            collisions.append(intersection)
            travelSegment = np.array([travel[POINT], intersection-travel[POINT]])
            w = getLineConst(travelSegment, pq)
            if(w > 0 and w < 1): numCrosses += 1
            travel[0] = intersection
            if side%2:
                travel[VECT][X] *= -1
            else:
                travel[VECT][Y] *= -1

assert(len(collisions) == m+n)
assert(any(np.all(np.equal(line[POINT], (collisions[-1]))) for line in table))          

for c in collisions:
    print c
    
print 'NumCrosses: ' + str(numCrosses)

p1 = np.array([0,0])
p2 = np.array([3,0])
p3 = np.array([3,4])

print getArea((p1, p2, p3))