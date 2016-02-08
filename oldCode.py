# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 09:20:17 2016

@author: lvanhulle
"""

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
    
#test to see if line pq goes beyond the table
#does not check if pq starts outside of the table
for side in table:
    t = getIntersectionConst(pq, side)
    if t > 0 and t < 1:
        pq[VECT] *= t