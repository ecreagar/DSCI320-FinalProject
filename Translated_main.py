import numpy as np
from numpy import linalg as linalg
from numpy.random import Generator, PCG64
from numpy.linalg import norm
from math import sqrt

import matplotlib.pyplot as plt

# s_1 = speed between A & P_1 = P_6 & B = s_7
# s_0 is a holder space
SPEEDS = [0,10,9,8,7,6,5,10]

# Takes the y vals for points 1-6
# Outputs the set of points
def get_points(Ys):
    if len(Ys) != 6:
        raise ValueError("Number of y values was not 6")

    # The common value of x i.e. P_1's x
    common = 25*sqrt(2)-25
    ret = [[0,0]] + [[common+10*i,Ys[i]] for i in range(6)] + [[50*sqrt(2)]*2]

    return np.array(ret)

# y incrament for line from A to B
y_inc = 25*(sqrt(2)-1)
# Starting ys (straight line from A to B)
ys_start = [y_inc*(i+1) for i in range(6)]
# Starting points
points_start = get_points(ys_start)

# Take the distance between each point and divide it by the speed in that region
# and sum it
# Takes the set of points
# Returns time it took
def obj_func(points):
    summed = 0
    for i in range(1,len(points)):
        summed += (1/SPPEDS[i])*norm(points[i]-points[i-1])**2

    return summed

# Takes the set of points
# Returns the slope for each y_i
# Note: y_0 and y_7 will have 0 since A and B are constants
def grad_obj_func(points):
    # We only need the y vals for each point since the xs never change
    ys = points[:,1]

    ret = np.zeros((8,1))

    for i in range(1,7):
        ret[i] = (2/SPEEDS[i])*(ys[i]-ys[i-1])-(2/SPEEDS[i+1])*(ys[i+1]-ys[i])

    return ret


















