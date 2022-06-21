import numpy as np
import queue
import matplotlib.pyplot as plt

NOISE = 0
UNASSIGNED = 0
core = -1
edge = -2

def DbSCAN_for_activations(categories, image_size):
    eps = 1
    minpts = 5
    counter = 0

    # for category in categories:
    eps += 5
    train = np.array(categories[-1])

    pointlabel, clusterNumber = dbscan(train, eps, minpts)
    counter += 1
    masks = plotRes(train, pointlabel, clusterNumber, counter, image_size)

    print('number of cluster found: ' + str(clusterNumber - 1))
    outliers = pointlabel.count(0)
    print('number of outliers found: ' + str(outliers) + '\n')
    return masks


# function to find all neigbor points in radius
def neighbor_points(data, pointId, radius):
    points = []
    for i in range(len(data)):
        if np.linalg.norm(data[i] - data[pointId]) <= radius:
            points.append(i)
    return points


# DB Scan algorithom
def dbscan(data, Eps, MinPt):
    pointlabel = [UNASSIGNED] * len(data)
    pointcount = []
    corepoint = []
    noncore = []

    # Find all neigbor for all point
    for i in range(len(data)):
        pointcount.append(neighbor_points(data, i, Eps))

    # Find all core point, edgepoint and noise
    for i in range(len(pointcount)):
        if (len(pointcount[i]) >= MinPt):
            pointlabel[i] = core
            corepoint.append(i)
        else:
            noncore.append(i)

    for i in noncore:
        for j in pointcount[i]:
            if j in corepoint:
                pointlabel[i] = edge

                break

    # start assigning point to cluster
    cl = 1
    # Using a Queue to put all neigbour core point in queue and find neigboir's neigbor
    for i in range(len(pointlabel)):
        q = queue.Queue()
        if (pointlabel[i] == core):
            pointlabel[i] = cl
            for x in pointcount[i]:
                if (pointlabel[x] == core):
                    q.put(x)
                    pointlabel[x] = cl
                elif (pointlabel[x] == edge):
                    pointlabel[x] = cl
            # Stop when all point in Queue has been checked
            while not q.empty():
                neighbors = pointcount[q.get()]
                for y in neighbors:
                    if (pointlabel[y] == core):
                        pointlabel[y] = cl
                        q.put(y)
                    if (pointlabel[y] == edge):
                        pointlabel[y] = cl
            cl = cl + 1  # move to next cluster

    return pointlabel, cl


# Function to plot final result
def plotRes(data, clusterRes, clusterNum, b, image_size):
    nPoints = len(data)
    scatterColors = [plt.cm.Spectral(each)
                     for each in np.linspace(0, 1, clusterNum + 1)]
    masks = []
    for i in range(clusterNum):
        if (i == 0):
            # Plot all noise point as blue
            color = 'blue'
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        mask = np.zeros(shape=(1, image_size, image_size, 3))

        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
                mask[0, image_size - data[j, 1], data[j, 0]] = [1, 1, 1]

        masks.append(mask)

        # plt.scatter(x1, y1, c=color, alpha=1, marker=',', s=1)
        # plt.show()
    return masks

