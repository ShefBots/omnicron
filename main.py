import numpy as np
from matplotlib import pyplot as plt
from random import random
import math

IMAGE_SIZE = 30 # The image is square.
SAMPLES = 100
MIRROR_SCALE = 0.95 # Radius of the mirror in the image in relation to the half-width of the image size.

class BasicNetwork:
    weights = np.random.rand(IMAGE_SIZE*IMAGE_SIZE) - 0.5
    def __init__(self, lr):
        self.learningRate = lr
    def forwardProp(self, data):
        flat = data.flatten()
        return (np.dot(flat, self.weights))

    def calculateError(actual, target):
        return (actual - target)

    def backProp(self, data, error):
        delta = -error * self.weights
        self.weights = self.weights - delta*self.learningRate

def main():
    data, labels = generateBasicData()
    #for i in range (int(SAMPLES/4)):
    #    plt.imshow(data[i], cmap="Greys_r");
    #    plt.show()
    bn = BasicNetwork(0.0001)
    for i in range(SAMPLES):
        actual = bn.forwardProp(data[i])
        target = labels[i]
        error = BasicNetwork.calculateError(actual, target)
        bn.backProp(data[i], error)
        print("[%i] Result: %.2f\tTarget: %.2f\tError: %.2f"%(i, actual, target, error))

# A simple data generator that just generates a line from the centerpoint outwards.
def generateBasicData():
    data = np.zeros((SAMPLES, IMAGE_SIZE, IMAGE_SIZE))
    labels = np.zeros(SAMPLES)
    midpoint = IMAGE_SIZE/float(2)
    radius = MIRROR_SCALE * midpoint
    for i in range(SAMPLES):
        angle = random() * math.pi * 2
        drawLine(data[i], int(midpoint), int(midpoint), int(midpoint + math.cos(angle)*radius), int(midpoint + math.sin(angle)*radius))
        labels[i] = angle * 180/math.pi
    return (data, labels)

def drawLine(matrix, x0, y0, x1, y1, color=255):
    xMin = min(x0,x1)
    xMax = max(x0,x1)
    if(abs(x0-x1) > abs(y0-y1)):
        # The x axis difference is bigger
        if(x0 > x1):
            # It does not ascend properly, swap 'em on the x.
            temp = x0
            x0 = x1
            x1 = temp
            temp = y0
            y0 = y1
            y1 = temp
        xdiff = x1-x0
        ydiff = y1-y0
        for x in range(x1-x0):
            y = y0+int((float(x)/xdiff)*ydiff)
            matrix[y, x0+x] = color
    else:
        # The y axis difference is bigger
        if(y0 > y1):
            # It does not ascend (but left?) properly, so flip 'em on the y.
            temp = y0
            y0 = y1
            y1 = temp
            temp = x0
            x0 = x1
            x1 = temp
        ydiff = y1-y0
        xdiff = x1-x0
        for y in range(y1-y0):
            x = x0+int((float(y)/ydiff)*xdiff)
            matrix[y0+y, x] = color

main();
