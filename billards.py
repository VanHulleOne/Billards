# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:37:59 2016

@author: lvanhulle
"""

import numpy as np
import time
import matplotlib.pyplot as plt

POINT, VECT = 0, 1
X, Y = 0,1
EPSILON = 0.001
ANGLE_EPS = np.cos(np.pi/2.0-.001) #0.001 radians (0.06 degrees) tolerance for perpendicular lines

m, n = 3, 5
p = np.array([.2, .1], float)
q = np.array([2.7, 4.97], float)
pq = np.array([p, q-p])

#lines are of the form startPoint, vector from start to end point
table = np.array([[[0,0], [m,0]], [[m,0],[0,n]], [[m,n], [-m,0]], [[0,n],[0,-n]]], float)

numCrosses = 0
collisions = [np.array([0,0], float)]
travel = np.array([[0,0], [1,1]], float)
intersection = np.array([0,0], float)
crossings = []

def hasArea(points):
    if len(points) != 3: raise Exception('getArea(points) Illegal number of points. \
                                Must use exactly 3 points.')
    matrix = np.ones([3,3])
    for i in xrange(len(points)):
        matrix[i][:2] = points[i]
        
    return abs(np.linalg.det(matrix)) > EPSILON
    
def areColinear(line1, line2):
    p2 = line1[POINT] + line1[VECT]
    p4 = line2[POINT] + line2[VECT]
    return not (hasArea((line1[POINT], p2, line2[POINT])) or
            hasArea((line1[POINT], p2, p4)))

def areParallel(line1, line2):
    perpVect = np.array([-line1[VECT][Y], line1[VECT][X]])
    #Farin-Hansford eq 3.14
    cosTheda = (np.dot(perpVect, line2[VECT])/
                (np.linalg.norm(perpVect)*np.linalg.norm(line2[VECT])))
    return abs(cosTheda) < ANGLE_EPS

def getLineConst(line, cLine):
    if(areParallel(line, cLine)):
        return None
    return (np.cross((line[POINT] - cLine[POINT]), line[VECT])/
            (1.0*np.cross(cLine[VECT], (line[VECT]))))
        
def getTU(uLine, tLine):
    return getLineConst(uLine, tLine), getLineConst(tLine, uLine)

def linePlot(line, style):
    plt.plot([line[POINT][X], line[POINT][X]+line[VECT][X]], [line[POINT][Y],
          line[POINT][Y]+line[VECT][Y]], style)

def isOnLine(line, t):
    t_Epsilon = EPSILON/np.linalg.norm(line[VECT])
    return not(t < t_Epsilon or t > 1-t_Epsilon)
    
def gcd(a, b):
    t = 0
    while(b != 0):
        t = a
        a = b
        b = t%b
    return a
    
#test to see if line pq goes beyond the table
#does not check if pq starts outside of the table
for side in table:
    t = getLineConst(side, pq)
    if t > 0 and t < 1:
        pq[VECT] *= t

for j in xrange((m+n)/gcd(m,n)-1):
    for side in xrange(len(table)):
        t, u = getTU(travel, table[side])
        if t >= 0 and t <= 1 and u > 0:
            intersection = table[side][POINT] + t * table[side][VECT]
            collisions.append(intersection)
            
            travelSegment = np.array([travel[POINT], intersection-travel[POINT]])
            w = getLineConst(travelSegment, pq)
            if(not(w is None) and isOnLine(pq, w)):
                numCrosses += 1
                crossings.append(np.array(pq[POINT] + w*pq[VECT]))
                
            travel[0] = intersection
            #if left or right side of table mirror about Y else mirror about X
            if side%2:
                travel[VECT][X] *= -1
            else:
                travel[VECT][Y] *= -1

print 'Collison Points:'
for c in collisions:
    print c
    
print '\nNumCrosses: ' + str(numCrosses)
for c in crossings:
    print c

assert(len(collisions) == (m+n)/gcd(m,n))
assert(any(np.all(np.equal(line[POINT], (collisions[-1]))) for line in table))          


p1 = np.array([2,1])
p2 = np.array([1.5,2])
p3 = np.array([3,4])

collisions = np.asarray(collisions)
crossings = np.asarray(crossings)
plt.axis([-m/10.0, m+m/10.0, -n/10.0, n+n/10.0])

for side in table:
    linePlot(side, 'k-')

plt.plot(collisions[:,0], collisions[:,1], 'b-')
linePlot(pq, 'g-')

if len(crossings) > 0:
    plt.plot(crossings[:,0], crossings[:,1], 'ro')
