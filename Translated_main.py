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
# Starting ys (straight line from A to B) -> slope = 1
ys_start = [y_inc+10*i for i in range(6)]
# Starting points
points_start = get_points(ys_start)

# Take the distance between each point and divide it by the speed in that region
# and sum it
# Takes the set of points
# Returns time it took
def obj_func(points):
    summed = 0
    for i in range(1,len(points)):
        summed += (1/SPEEDS[i])*norm(points[i]-points[i-1])

    return summed

# Takes the set of points
# Returns the slope for each y_i
# Note: y_0 and y_7 will have 0 since A and B are constants
def grad_obj_func(points):
    ret = np.zeros((8,1))

    for i in range(1,7):
        # when i=i
        ret_1 = -(points[i-1,1]-points[i,1]) / (SPEEDS[i]*norm(points[i-1]-points[i]))
        # when i=i+1
        ret_2 = (points[i,1]-points[i+1,1]) / (SPEEDS[i+1]*norm(points[i]-points[i+1]))

        ret[i,0] = ret_1+ret_2

    return ret

# Steepest Descent
################################################################################
class Descent():

    # f_x is the obj func || grad is the 1st grad of the obj func
    # x0 is the starting point
    # x0 has 8 points in the form [x,y] -> shape=(8,2)
    def __init__(self,f_x,grad,x0):
        self.f_x = f_x
        self.grad = grad
        self.x_ks = [x0]

        # Iteration number
        self.k = 0

        # Set of all p_ks
        self.p_ks = []

        self.step()

    def step(self):
        # p_k direction to move
        p_k = -self.grad(self.x_ks[-1]).T
        self.p_ks.append(p_k)

        # I choose to use a back tracking method to find the best a_k
        a_k = self.get_alpha(p_k)

        # Calculate the new point
        # Note: I am only changing (:,1) because we only need to change the ys
        x_k1 = self.x_ks[-1]
        x_k1[:,1] = self.x_ks[-1][:,1]+a_k*p_k
        # Add the new point
        self.x_ks.append(x_k1)

        self.k += 1

        return x_k1

    # Uses a binary method for back tracking line search to find the best alpha_k
    # I fist use alpha as "start" and "end", find which has the lowest obj func value,
    # then find the midpoint
    def get_alpha(self, p_k):
        start, end = 1e-7, 1-(1e-7)
        for i in range(20):
            # x_k value using "start" as alpha
            x_start = self.x_ks[-1][:,1]+start*p_k
            # x_k value using "end" as alpha
            x_end = self.x_ks[-1][:,1]+end*p_k

            # Get the respective obj func values
            f_start = self.f_x(x_start)
            f_end = self.f_x(x_end)

            # Binary Choice
            if f_start < f_end:
                end = (end+start)/2
            else:
                start = (end+start)/2

        return (end+start)/2


# Run the algorithm and get approite data
################################################################################
# Correct answer = 13.1265108586
def main(ERR=1e-7):
    SGD = Descent(obj_func, grad_obj_func, points_start.copy())

    # Gradient length from the last x_k
    grad_norm = norm(SGD.grad(SGD.x_ks[-1]))

    # Run algorithm till the derviative is within the ERR to 0
    while grad_norm > ERR:
        SGD.step()
        grad_norm = norm(SGD.grad(SGD.x_ks[-1]))

    return SGD














