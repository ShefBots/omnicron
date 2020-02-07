import numpy as np
from matplotlib import pyplot as plt
from random import random
import math

IMAGE_SIZE = 15 # The image is square.
SAMPLES = 100 # Batch
REPEATS = 100000
#SAMPLES = 10000 # Single
#REPEATS = 1
LR = 0.00000001
MIRROR_SCALE = 0.95 # Radius of the mirror in the image in relation to the half-width of the image size.

# A single-layer, single-output network
class BasicNetwork:
    def __init__(self, lr):
        self.learningRate = lr
        self.weights = np.random.rand(IMAGE_SIZE*IMAGE_SIZE) - 0.5
        self.bias = np.random.random()
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

    def backProp(self, data, actual, target, debug=False):
        flat = data.flatten()
        delta = (target - actual) * flat # The differential of the error fn wrt weights (div. 2)
        deltaB = (target - actual) # [* 1]  # The differential of the error fn wrt the bias.
        if(debug):
            return (delta, deltaB) # Return deltas for debug
        self.weights = self.weights + delta*self.learningRate
        self.bias = self.bias + deltaB*self.learningRate

    def backPropBatch(self, data, actual, target, debug=False):
        # Reshape to be a batch of 1D images
        images = data.reshape((data.shape[0], IMAGE_SIZE*IMAGE_SIZE))
        differences1D = target-actual # Calculate the differences between the targets
        differences = np.diag(differences1D) # Store the differences in a matrix to use them to scale the weights
        # The PER-WEIGHT (PIXEL) AVERAGE differential of the error fn wrt weights (div. 2) across all images in batch:
        delta = (differences.dot(images)).mean(axis=0)
        deltaB = differences1D.mean() # [* 1]  # The AVERAGE differential of the error fn wrt the bias.
        if(debug):
            return (delta, deltaB) # Return deltas for debug
        self.weights = self.weights + delta*self.learningRate
        self.bias = self.bias + deltaB*self.learningRate

    def plotInfo(self, errors):
        plt.plot(errors)
        plt.show()
        plt.imshow(self.weights.reshape((IMAGE_SIZE,IMAGE_SIZE)))
        plt.show()

def main():
    batchTrain()
    #onlineTrain()
    #test()

def test():
    # Hack the image size smaller for easier reading
    global IMAGE_SIZE
    IMAGE_SIZE = 4

    # Generate results, compare the backprop implementations
    data, labels = generateBasicData(1)
    bn = BasicNetwork(0.0000001)
    actual = bn.forwardProp(data) # This is vector of results (one for each sample)
    error = BasicNetwork.calculateError(actual, labels) # This is vector of errors

    # TESTING BACKPROP METHODS:
    target = labels[0]

    # SINGULAR BACKPROP
    delta, deltaB = bn.backProp(data, actual, target, debug=True)
    print("singular delta:", delta)
    print("singular deltab:", deltaB)

    # BATCH BACKPROP
    delta, deltaB = bn.backPropBatch(data, actual, target, debug=True)
    print("batch delta:", delta)
    print("batch deltab:", deltaB)

def batchTrain():
  bn = BasicNetwork(LR)
  errors = np.zeros(REPEATS)
  for i in range(REPEATS):
    data, labels = generateBasicData(SAMPLES)
    actual = bn.forwardProp(data) # This is vector of results (one for each sample)
    error = BasicNetwork.calculateError(actual, labels) # This is vector of errors
    bn.backPropBatch(data, actual, labels)
    meanError = error.mean()
    print("[%i] %.2f pct\tResult: %.2f\tTarget: %.2f\tError: %.2f"%(i, float(i)/float(REPEATS)*100, actual[0], labels[0], meanError))
    errors[i] = meanError
    bn.learningRate *= 0.9999999999999
  bn.plotInfo(errors)

def onlineTrain():
    data, labels = generateBasicData(SAMPLES)
    #for i in range (int(min(SAMPLES/4, 10)):
    #    plt.imshow(data[i], cmap="Greys_r");
    #    plt.show()
    bn = BasicNetwork(LR)
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
