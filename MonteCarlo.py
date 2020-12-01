#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


speeds = [10, 9, 8, 7, 6, 5]


def plotProblem(points, duration):
	A = [0,50]
	B = [100,50]
	plt.scatter(A[0], A[1], color="blue")
	plt.text(A[0]+1,A[1]+1,"A")
	plt.scatter(B[0], B[1], color='blue')
	plt.text(B[0]+1,B[1]+1,"B")
	x = np.linspace(0,100,1000)
	plt.plot(x, x+25*np.sqrt(2), linestyle='--', color="yellow")
	plt.plot(x, x+15*np.sqrt(2), linestyle='--', color="orange")
	plt.plot(x, x+5*np.sqrt(2), linestyle='--', color="orangered")
	plt.plot(x, x-5*np.sqrt(2), linestyle='--', color="orangered")
	plt.plot(x, x-15*np.sqrt(2), linestyle='--', color="red")
	plt.plot(x, x-25*np.sqrt(2), linestyle='--', color="darkred")

	if (points != None):
		plt.plot([i[0] for i in points], [j[1] for j in points], linestyle=":", 
			label = str(duration)+" Days")
		plt.scatter([i[0] for i in points], [j[1] for j in points])
	plt.legend()
	plt.xlim([-2,102])
	plt.ylim([0,100])
	plt.show()


def distance(point1, point2):
	'''
	Takes two points and returns the distance between them
	'''
	return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def createPoints(x2,x3,x4,x5,x6,x7):
	point1 = [0,50]
	point8 = [100,50]
	point2 = [x2, x2+25*np.sqrt(2)]
	point3 = [x3, x3+15*np.sqrt(2)]
	point4 = [x4, x4+5*np.sqrt(2)]
	point5 = [x5, x5-5*np.sqrt(2)]
	point6 = [x6, x6-15*np.sqrt(2)]
	point7 = [x7, x7-25*np.sqrt(2)]

	points = [point1,point2,point3,point4,point5,point6,point7,point8]
	return(points)


def tripDuration(points):
	duration = 0
	for i in range(len(points)-1):
		duration += distance(points[i],points[i+1])/speeds[i%6]
	return duration


def MonteCarlo(points,iterations):
	# Start with point of straight line through the problem
	duration0 = tripDuration(createPoints(points[0],points[1],points[2],points[3],points[4],
			points[5]))
	minDuration = duration0
	bestPoints = points
	durations = [duration0]
	# Use a T value of 10
	T = 1
	# TODO: Use simulated annealing to allow us to shrink the variability and
	# be more precise?
	for i in range(iterations):
		# add randomness between -2 and 2 to points
		# and calculate the trip duration at these new points
		pointsNew = []
		for i in points:
			pointsNew.append(i + (np.random.rand()-0.5)*4)
		route = createPoints(pointsNew[0], pointsNew[1], pointsNew[2],
			pointsNew[3], pointsNew[4], pointsNew[5])
		duration = tripDuration(route)
		durations.append(duration)
		# decide whether to take the new points by comparing the trip duration 
		# to the last one
		if(duration <= minDuration):
			# the currect points give us our best value, save the value and 
			# points
			minDuration = duration
			points = pointsNew.copy()
		else:
			# if not, accept the new point with a small probability
			s = np.random.rand()
			if((np.exp(-1*(duration - minDuration)/T)) <= s):
				minDuration = duration
				points = pointsNew.copy()


	plt.plot(durations)
	plt.show()

	return points,duration


def main():
	# Show the original problem with a straght line
	starting = (100-5*(10*np.sqrt(2)))/2
	step = 10*np.sqrt(2)
	directxs = [starting, starting+step, starting+step*2, 
		starting+step*3, starting+step*4, starting+step*5]
	direct = createPoints(starting, starting+step, starting+step*2, 
		starting+step*3, starting+step*4, starting+step*5)
	print("Direct Route: ",tripDuration(direct),"days")
	duration = tripDuration(direct)
	plotProblem(direct,duration)

	MC_points,MC_duration = MonteCarlo(directxs,100000)
	MC_route = createPoints(MC_points[0],MC_points[1],MC_points[2],MC_points[3],
		MC_points[4],MC_points[5])
	plotProblem(MC_route,MC_duration)
	pass


if __name__ == '__main__':
	main()
