import numpy as np
from matplotlib import pyplot as plt
from random import random
import math

IMAGE_SIZE = 15 # The image is square.
SAMPLES = 1000
REPEATS = 100
MIRROR_SCALE = 0.95 # Radius of the mirror in the image in relation to the half-width of the image size.

# A single-layer, single-output network
class BasicNetwork:
    weights = np.random.rand(IMAGE_SIZE*IMAGE_SIZE) - 0.5
    bias = np.random.random()
    def __init__(self, lr):
        self.learningRate = lr
    def forwardProp(self, data):
        if(len(data.shape) == 3):
          newShape = (data.shape[0], IMAGE_SIZE*IMAGE_SIZE)
        else:
          newShape = (IMAGE_SIZE*IMAGE_SIZE)
        flat = data.reshape(newShape)
        return (np.dot(flat, self.weights) + self.bias)

    # Calculates the squared error on this output
    def calculateError(actual, target):
        return ((target - actual)**2)

    def backProp(self, data, actual, target):
        flat = data.flatten()
        delta = (target - actual) * flat # The differential of the error fn wrt weights (div. 2)
        self.weights = self.weights + delta*self.learningRate
        deltaB = (target - actual) # [* 1]  # The differential of the error fn wrt the bias.
        self.bias = self.bias + deltaB*self.learningRate

    def backPropBatch(self, data, actual, target):
        flat = data.reshape((data.shape[0], IMAGE_SIZE*IMAGE_SIZE)) # Reshape to be a batch
        print("TARGET:")
        print(target.shape)
        print("ACTUAL:")
        print(actual.shape)
        print("FLAT:")
        print(flat.shape)
        print(flat)
        sub = target - actual
        delta = ((target - actual) * flat).mean() # The AVERAGE differential of the error fn wrt weights (div. 2)
        self.weights = self.weights + delta*self.learningRate
        deltaB = (target - actual).mean() # [* 1]  # The AVERAGE differential of the error fn wrt the bias.
        self.bias = self.bias + deltaB*self.learningRate

    def plotInfo(self, errors):
        plt.plot(errors)
        plt.show()
        plt.imshow(self.weights.reshape((IMAGE_SIZE,IMAGE_SIZE)))
        plt.show()

def main():
    batchTrain()
    #onlineTrain()

def batchTrain():
  bn = BasicNetwork(0.0000001)
  errors = np.zeros(REPEATS)
  for i in range(REPEATS):
    data, labels = generateBasicData(SAMPLES)
    actual = bn.forwardProp(data) # This is vector of results (one for each sample)
    error = BasicNetwork.calculateError(actual, labels) # This is vector of errors
    bn.backPropBatch(data, actual, labels)
    meanError = error.mean()
    print("[%i] %.2f pct\tResult: %.2f\tTarget: %.2f\tError: %.2f"%(i, float(i)/float(SAMPLES*REPEATS)*100, actual[0], labels[0], meanError))
    errors[i] = meanError
  bn.plotInfo(errors)

def onlineTrain():
    data, labels = generateBasicData(SAMPLES)
    #for i in range (int(min(SAMPLES/4, 10)):
    #    plt.imshow(data[i], cmap="Greys_r");
    #    plt.show()
    bn = BasicNetwork(0.000000001)
    errors = np.zeros(SAMPLES*REPEATS)
    for i in range(SAMPLES*REPEATS):
        actual = bn.forwardProp(data[i%SAMPLES])
        target = labels[i%SAMPLES]
        error = BasicNetwork.calculateError(actual, target)
        bn.backProp(data[i%SAMPLES], actual, target)
        print("[%i] %.2f pct\tResult: %.2f\tTarget: %.2f\tError: %.2f"%(i, float(i)/float(SAMPLES*REPEATS)*100, actual, target, error))
        errors[i] = error
    bn.plotInfo(errors)

# A simple data generator that just generates a line from the centerpoint outwards.
def generateBasicData(count):
    data = np.zeros((count, IMAGE_SIZE, IMAGE_SIZE))
    labels = np.zeros(count)
    midpoint = IMAGE_SIZE/float(2)
    radius = MIRROR_SCALE * midpoint
    for i in range(count):
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
