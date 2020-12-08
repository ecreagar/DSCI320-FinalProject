#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


speeds = [10, 9, 8, 7, 6, 5]


def plotProblem(points, duration, *args, **kwargs):
    title = kwargs.get('title', None)
    # Plot points A and B
    A = [0, 50]
    B = [100, 50]
    plt.scatter(A[0], A[1], color="blue")
    plt.text(A[0]+1, A[1]+1, "A")
    plt.scatter(B[0], B[1], color='blue')
    plt.text(B[0]+1, B[1]+1, "B")

    # plot the lines in the marsh
    x = np.linspace(0, 100, 1000)
    plt.plot(x, x+25*np.sqrt(2), linestyle='--', color="yellow")
    plt.plot(x, x+15*np.sqrt(2), linestyle='--', color="orange")
    plt.plot(x, x+5*np.sqrt(2), linestyle='--', color="orangered")
    plt.plot(x, x-5*np.sqrt(2), linestyle='--', color="orangered")
    plt.plot(x, x-15*np.sqrt(2), linestyle='--', color="red")
    plt.plot(x, x-25*np.sqrt(2), linestyle='--', color="darkred")

    # Plot the points that Frodo and Sam walk through
    if (points is not None):
        plt.plot([i[0] for i in points], [j[1] for j in points], linestyle=":",
                 label=str(duration)+" Days")
        plt.scatter([i[0] for i in points], [j[1] for j in points])
    plt.legend()

    # Title optional
    if(title is None):
        plt.title("Frodo and Sam's best path to point B")
    if(title is not None):
        plt.title(title)

    plt.xlim([-2, 102])
    plt.ylim([0, 100])
    plt.show()


def distance(point1, point2):
    '''
    Takes two points and returns the distance between them
    '''
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def createPoints(points):
    '''
    Given the x values for our points, return the corresponding X,Y pairs
    which lie on the lines through the swamp
    '''
    point1 = [0, 50]
    point8 = [100, 50]
    point2 = [points[0], points[0]+25*np.sqrt(2)]
    point3 = [points[1], points[1]+15*np.sqrt(2)]
    point4 = [points[2], points[2]+5*np.sqrt(2)]
    point5 = [points[3], points[3]-5*np.sqrt(2)]
    point6 = [points[4], points[4]-15*np.sqrt(2)]
    point7 = [points[5], points[5]-25*np.sqrt(2)]

    points = [point1, point2, point3, point4, point5, point6, point7, point8]
    return(points)


def tripDuration(points):
    duration = 0
    for i in range(len(points)-1):
        duration += distance(points[i], points[i+1])/speeds[i % 6]
    return duration


# Create Generation
def createGen0(genSize):
    '''
    Our individuals are sets of 8 (could expand to more if we want) points
    corresponding to the x values at each different section of the marsh (and
    the starting and ending points, which don't change)
    '''
    # Pass random values to createPoints()
    inds = []

    for i in range(genSize):
        # pick 6 random integers between 1 and 100 to create points from
        inds.append(createPoints(np.random.randint(100, size=6).tolist()))

    return inds
    pass


def mySortKey(point):
    return(tripDuration(point))


def sort(points):
    points.sort(key=mySortKey, reverse=False)


def sortByFitness(points):
    '''
    The fitness of the individual is the time it takes it to get through the
    swamp (a.k.a. tripDuration).

    Pass in the individuals, and return them sorted by best fitness to worst.
    '''
    # sort the individuals by fitness
    sort(points)

    # return the fitnesses as well in case we want it for plotting
    return([tripDuration(i) for i in points])


def mate(individuals):
    '''
    Mating is defined as taking the average between each of the x values and
    returning the new points as the child.

    Pass in the individuals and return 5 children (x points only) who have
    been mated: individual 1 with ind. 2, 1 with 3 ... 1 with 6.
    I like this way of mating because it allows the best fitness to have more
    influence but also should allow for variability
    '''
    topIndividuals = individuals[0:6]
    children = []

    i = 1
    while(len(children) < 5):
        # individuals are [[[x,y],[x,y]],[[x,y][x,y]]] and so forth, so we want
        # to averate the 0th element from the 0th individual and
        # ith individual for each point j
        children.append([[(topIndividuals[0][j][0]+topIndividuals[i][j][0])/2]
                        for j in range(len(topIndividuals[0]))])
        i += 1

    return children


def mutate(children, genSize, randomness):
    '''
    Takes in the 5 children from mate and adds randomness to them.
    Randomly select one, add some randomness, make it into a point,
    repeat genSize times.
    '''
    individuals = []
    for i in range(genSize):
        # select a random child
        rand = np.random.randint(0, len(children))
        # createPoints only takes in the 2nd through 7th points
        # (not start or finish)
        tempChild = children[rand][1:-1]
        # Amount of randomness varies with randomness param. if
        # randomness is 1, the value added is rand between -1 and 1
        randomXs = [(tempChild[j][0] + ((np.random.rand()-0.5) * 2) *
                    randomness) for j in range(len(tempChild))]
        individuals.append(createPoints(randomXs))

    return individuals


def runGenetic(genSize, numGenerations):
    '''
    1. create generation 0
    2. evaluate their fitness
    3. mate the best individuals
    4. mutate the individuals to make the new generation size genSize
    5. repeat 2-4 numGenerations times
    '''
    # 1
    individuals = createGen0(genSize)

    minDuration = 1000
    topDog = []
    durationsOverTime = []
    # 2-4
    for i in range(numGenerations):
    # sorts the individuals and gives us a list of durations for plotting or
    # whatever
        durations = sortByFitness(individuals)
        durationsOverTime.append(durations[0])
        if(durations[0] < minDuration):
            minDuration = durations[0]
            topDog = individuals[0]

        children = mate(individuals)

        individuals = mutate(children, genSize=genSize, randomness=(10/(i+1)))

        if(np.abs(minDuration - 13.1265) < .0001):
            break

    plt.plot(durationsOverTime)
    plt.title("Trip Duration by Iteration")
    plt.ylabel("Trip Duration (Days)")
    plt.show()
    return topDog, minDuration, i


def main():

    # Show the original problem with a straght line
    starting = (100-5*(10*np.sqrt(2)))/2
    step = 10*np.sqrt(2)

    # create the steps for the original, straght line problem
    direct = createPoints([starting, starting+step, starting+step*2,
                           starting+step*3, starting+step*4, starting+step*5])
    print("Direct Route: ", tripDuration(direct), "days")
    duration = tripDuration(direct)

    # Plot the original problem
    plotProblem(direct, duration)
    
    bestPath, bestTime, numGens = runGenetic(genSize=100, numGenerations=100)

    plotProblem(bestPath, bestTime, title="Genetic Algorithm - Frodo and Sam's best path")
    print("Genetic Algorithm: ", bestTime, "days")
    print("Converged in: ", numGens, "generations")
    pass


if __name__ == '__main__':
    main()
