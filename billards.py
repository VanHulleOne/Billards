# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:37:59 2016

@author: lvanhulle
"""

import numpy as np
import time
import matplotlib.pyplot as plt


#Lines are stored in the form [start point, vector to end point]
#These constants call the start point (POINT) or vector (VECT)
POINT, VECT = 0, 1
#Constants for the X and Y of a point/vector
X, Y = 0,1
#Error amount used for distance checks (or EPSILON^2 for area checks)
EPSILON = 0.001
#0.001 radians (0.06 degrees) tolerance for perpendicular lines
ANGLE_EPS = np.cos(np.pi/2.0-.001) 

"""
The following three variables are for user input.

m, n are the lengths of the table sides. From the problem they must
be co-prime integers.

The two points p, q are the start and end points of the line of the table.
They are stored in Numpy arrays of type float and are of the form [x, y]
p = start point of the line on the table in the form [x, y]
q = end point of the line on the table als [x, y]

"""
m, n = 3, 7 #Lengths of the table sides
p = np.array([.2, .1], float) #start point of line pq
q = np.array([2.7, 4.97], float) #end point of line pq


pq = np.array([p, q-p]) #the line pq in the form used in this program

def gcd(a, b):
    """
    gcd takes in two numbers, makes sure they are both integers and then
    determines their greatest common denominator using the Euclid method.
    """
    if not isinstance(a+b, int):
        raise Exception('User input error. Variables m and n must be integers.')
    t = 0
    while(b != 0):
        t = a
        a = b
        b = t%b
    return a

gcd = gcd(m,n) #determine the gcd of m and n
#if the side lengths are not co-prime raise an Exception
if gcd != 1:
    raise Exception(('Invalid table dimensions (m, n). %d and %d are not ' + 
                        'co-prime. GCD = %d') %(m, n, gcd))

#lines are of the form [startPoint, vector from start to end point]
#the Numpy array storing the sides of the table
table = np.array([[[0,0], [m,0]], [[m,0],[0,n]],
                  [[m,n], [-m,0]], [[0,n],[0,-n]]], float)

numCrosses = 0 #Accumulates the number of times the line pq is crossed
collisions = [np.array([0,0], float)] #Stores the points where the ball hits the edges of the table
travel = np.array([[0,0], [1,1]], float) #the current travel path of the ball
intersection = np.array([0,0], float) #The current place where the ball hit the bumper
crossings = [] #the points where the ball crosses pq

def hasArea(points):
    """
    This method takes in a list of three points and determines if they form
    a triangle with an area greater than EPSILON. If the results is False
    then we know that the points are co-linear. The test is performed by
    placing the three points into a 3x3 matrix with 1's in the third column
    and finding the determinant of that matrix (which is actually x2 the area).
    """
    #Make sure the list has three points
    if len(points) != 3:
        raise Exception('getArea(points) Illegal number of ' + 
                        'points. Must use exactly 3 points.')
                        
    matrix = np.ones([3,3]) #initialize the 3x3 matrix with 1's
    for i in xrange(len(points)):
        #for each point in the list place its point and vector into the matrix
        matrix[i][:2] = points[i]
    #If the abs of the area of the triangle is greater the EPSILON^2 return True
    return abs(np.linalg.det(matrix)) > EPSILON**2
    
def areColinear(line1, line2):
    """
    areColinear takes in 
    """
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

def isOnLineSegment(line, t):
    t_Epsilon = EPSILON/np.linalg.norm(line[VECT])
    return not(t < t_Epsilon or t > 1-t_Epsilon)
    

    
#test to see if line pq goes beyond the table
#does not check if pq starts outside of the table
for side in table:
    t = getLineConst(side, pq)
    if t > 0 and t < 1:
        pq[VECT] *= t

for j in xrange(m+n-1):
    for side in xrange(len(table)):
        t, u = getTU(travel, table[side])
        if t >= 0 and t <= 1 and u > 0:
            intersection = table[side][POINT] + t * table[side][VECT]
            collisions.append(intersection)
            
            travelSegment = np.array([travel[POINT], intersection-travel[POINT]])
            w = getLineConst(travelSegment, pq)
            if(not(w is None) and isOnLineSegment(pq, w)):
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

assert(len(collisions) == (m+n))
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
