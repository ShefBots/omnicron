# Robot ball bearing height is 185.5mm

print("Loading libraries...")
#from picamera.array import PiRGBArray
#from picamera import PiCamera
#import time
import cv2
import math
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os

import random

print("Loading dewarp data...")
import Dewarper
print("Dewarp data loaded!")

MINIMUM_BLOB_SIZE = 80
MM_PER_PIXEL = 10
ROBOT_VIEW_DIR_LEN = 5
SEARCH_RADIUS = 40

FRAMES_PATH = "../data/strafeSquare"
FILE_STRING = "frame%04d.png"
ROTATION = False

LOWER_RED = np.array([90, 200, 0])
UPPER_RED = np.array([130, 255, 255])
LOWER_GREEN = np.array([10, 100, 0])
UPPER_GREEN = np.array([100, 255, 255])
## Interestingly, the below seems to detect almost all barrels:
#LOWER_ALL = np.array([0, 200, 0])
#UPPER_ALL = np.array([360, 255, 255])

# Helper lambda functions
normalizeMatrix = lambda m: m/np.max(m)
rescaleMatrix = lambda m: normalizeMatrix(m)*255
intTuple = lambda t: (int(t[0]), int(t[1]))
dewarpedDims = Dewarper.dewarpData['lookup-table'].shape
downscale = lambda x: x/MM_PER_PIXEL
upscale = lambda x: x*MM_PER_PIXEL


worldMap = np.zeros((int(downscale(6000)), int(downscale(6000)))) # World is 3000x3000mm (3x3m)
robotPosition = (worldMap.shape[0]/2, worldMap.shape[1]/2)
robotOrientation = 0
robotCoordinates = []

class Barrel:
  BARREL_HEIGHT = 80
  BARREL_RADIUS = 10
  ROBOT_EYE_HEIGHT = 185.5

  eyeHeightAboveBarrel = ROBOT_EYE_HEIGHT - BARREL_HEIGHT

  def __init__(self, x, y, width, robotOrientation):
    viewCoords = (x + int(width/2), y)
    self.angle = float(viewCoords[0])/Dewarper.dewarpData['lookup-table'].shape[0] * math.pi*2 + robotOrientation
    self.distance = math.tan(Dewarper.dewarpData['angles'][y]) * Barrel.eyeHeightAboveBarrel
    self.generateKernelCoords()

  def generateKernelCoords(self):
    relativeCoords = np.asarray([math.cos(self.angle) * self.distance, math.sin(self.angle) * self.distance]) 
    self.kernelSpaceCoords = downscale(relativeCoords)

  def getAngle(self):
    return self.angle

  def setAngle(self, angle):
    self.angle = angle
    self.generateKernelCoords()

  def rotate(self, angle):
    self.angle += angle
    self.generateKernelCoords()

  def drawToKernel(self, m, offset):
    cv2.circle(m, (int(offset[0] + self.kernelSpaceCoords[0]), int(offset[1] + self.kernelSpaceCoords[1])), int(downscale(Barrel.BARREL_RADIUS)), 1.0, -1)

  def __str__(self):
    return("Distance: %.1f"%(self.distance) + "\nAngle: %.1f"%(self.angle))

class BarrelCollection:
  EDGE_OFFSET = np.ones(2) * downscale(Barrel.BARREL_RADIUS)

  def __init__(self, listOfBarrels):
    self.minBounds = np.zeros(2, dtype=np.float16) # [x min, y min]
    self.maxBounds = np.zeros(2, dtype=np.float16) # [x max, y max]

    #TODO: Okay, we loop through them here again instead. This could be overhauled with the addition of single-shot filtering
    for b in listOfBarrels:
      # Track the min and max bounds
      self.maxBounds = np.maximum(self.maxBounds, b.kernelSpaceCoords)
      self.minBounds = np.minimum(self.minBounds, b.kernelSpaceCoords)

    # Expand to allow space for edges
    self.minBounds -= BarrelCollection.EDGE_OFFSET
    self.maxBounds += BarrelCollection.EDGE_OFFSET

    # At the moment the coordinates are very simple, but in the future padding will be added for blurring
    self.robotCoordinates = (-self.minBounds[0], -self.minBounds[1]) # The coordinates to go from top left to robot location
    self.kernelSize = (int(self.maxBounds[0]-self.minBounds[0]), int(self.maxBounds[1]-self.minBounds[1]))
    self.kernel = np.zeros((self.kernelSize[1], self.kernelSize[0]))
    self.tl = (0,0) # Default top-left position of this kernel
    for b in listOfBarrels:
      b.drawToKernel(self.kernel, self.robotCoordinates)

  def matchToLocalWorld(self, worldMatrix, robotPosition):
    # Convolve the kernel over the map at different angles to find it's most likely position and orientation

    # Crop a submatrix from the worldMatrix to search
    searchDistance = int(downscale(100)) # Will search around the robot +-100mm
    #cropTL = (int(robotPosition[1] + self.minBounds[0] - searchDistance), int(robotPosition[0] + self.minBounds[1] - searchDistance))
    #cropBR = (int(robotPosition[1] + self.maxBounds[0] + searchDistance), int(robotPosition[0] + self.minBounds[1] + searchDistance))
    cropTL = (int(robotPosition[1] + self.minBounds[0] - searchDistance), int(robotPosition[0] + self.minBounds[1]))
    cropBR = (int(robotPosition[1] + self.maxBounds[0] + searchDistance), int(robotPosition[0] + self.maxBounds[1]))
    worldMatrixCrop = worldMatrix[cropTL[0]: cropBR[0], cropTL[1] : cropBR[1]]
    print("Inner Crop: ", worldMatrixCrop.shape)
    print("Inner Crop Pos: ", cropTL, " to ", cropBR)
    print("kernel Size: ", self.kernel.shape)
    print("TL: ", self.tl)


    likelihoodMap = cv2.filter2D(worldMatrixCrop, cv2.CV_32F, self.kernel)
    probablePosition = np.argmax(likelihoodMap)
    probablePosition = (probablePosition%likelihoodMap.shape[1], math.ceil(probablePosition/likelihoodMap.shape[1]))
    self.tl = intTuple((probablePosition[0]-self.kernelSize[0]/2+cropTL[0], probablePosition[1]-self.kernelSize[1]/2+cropTL[1]))
    worldPos = (self.tl[0]+self.robotCoordinates[0], self.tl[1]+self.robotCoordinates[1])

    # Now, rotate the kernel and try again - remember to rotate the robot coordinates

    ##### DRAWING ####
    #likelihoodMapDisplay = rescaleMatrix(likelihoodMap.copy())
    #br = (self.tl[0] + self.kernelSize[0], self.tl[1] + self.kernelSize[1])
    #cv2.rectangle(likelihoodMapDisplay, intTuple(self.tl), intTuple(br), 128, 1)
    #cv2.circle(likelihoodMapDisplay, intTuple(worldPos), int(downscale(Barrel.BARREL_RADIUS/2)), 128, -1)
    #cv2.imwrite("output/barrelKernels/"+filename.strip(".png")+"-b-likelihoodMap.png", likelihoodMapDisplay)
    ##################

    # TODO: Don't call max() here, just use the position from argmax.
    # Also we can totally combine all of these convolutions into one convolution with N layers
    return (worldPos, likelihoodMap.max())

  def matchToWorld(self, worldMatrix, robotPosition):
    # Convolve the kernel over the map at different angles to find it's most likely position and orientation
    # (In practice this will only convolve initially within a region of the robot's position and within a small
    # range of rotations, here we will convolve over the entire map)
    likelihoodMap = cv2.filter2D(worldMatrix, cv2.CV_32F, self.kernel)

    ## Hide all the bits that are too far away (It now can't move more than 200mm
    # Calculate Seach Bounds
    searchArea = int(downscale(SEARCH_RADIUS))
    kernelRobotPosition = (robotPosition[0]-self.robotCoordinates[0]+self.kernelSize[0]/2, robotPosition[1]-self.robotCoordinates[1]+self.kernelSize[1]/2)
    tls = intTuple((kernelRobotPosition[0] - searchArea, kernelRobotPosition[1] - searchArea))
    brs = intTuple((kernelRobotPosition[0] + searchArea, kernelRobotPosition[1] + searchArea))

    # Fill non-search bounds
    cv2.rectangle(likelihoodMap, (0,0), (likelihoodMap.shape[1], tls[1]), 0, -1)
    cv2.rectangle(likelihoodMap, (0,brs[1]), (likelihoodMap.shape[1], likelihoodMap.shape[0]), 0, -1)
    cv2.rectangle(likelihoodMap, (0,tls[1]), (tls[0], brs[1]), 0, -1)
    cv2.rectangle(likelihoodMap, (brs[0], tls[1]), (likelihoodMap.shape[1], brs[1]), 0, -1)

    # Continue as normal
    probablePosition = np.argmax(likelihoodMap)
    probablePosition = (probablePosition%likelihoodMap.shape[1], math.ceil(probablePosition/likelihoodMap.shape[1]))


    self.tl = intTuple((probablePosition[0]-self.kernelSize[0]/2, probablePosition[1]-self.kernelSize[1]/2))
    worldPos = (self.tl[0]+self.robotCoordinates[0], self.tl[1]+self.robotCoordinates[1])

    likelihoodMapDisplay = rescaleMatrix(likelihoodMap.copy())
    #br = (self.tl[0] + self.kernelSize[0], self.tl[1] + self.kernelSize[1])
    #cv2.rectangle(likelihoodMapDisplay, intTuple(self.tl), intTuple(br), 128, 1)
    #cv2.circle(likelihoodMapDisplay, intTuple(worldPos), int(downscale(Barrel.BARREL_RADIUS/2)), 128, -1)
    #cv2.rectangle(likelihoodMapDisplay, tls, brs, 255, 1)
    #cv2.imwrite("output/barrelKernels/"+filename.strip(".png")+"-b-likelihoodMap.png", likelihoodMapDisplay)
    return (worldPos, likelihoodMap.max())

  def matchToWorldClassic(self, worldMatrix, robotPosition):
    # Convolve the kernel over the map at different angles to find it's most likely position and orientation
    # (In practice this will only convolve initially within a region of the robot's position and within a small
    # range of rotations, here we will convolve over the entire map)
    likelihoodMap = cv2.filter2D(worldMatrix, cv2.CV_32F, self.kernel)
    probablePosition = np.argmax(likelihoodMap)
    probablePosition = (probablePosition%likelihoodMap.shape[1], math.ceil(probablePosition/likelihoodMap.shape[1]))
    self.tl = intTuple((probablePosition[0]-self.kernelSize[0]/2, probablePosition[1]-self.kernelSize[1]/2))
    worldPos = (self.tl[0]+self.robotCoordinates[0], self.tl[1]+self.robotCoordinates[1])
    return (worldPos, likelihoodMap.max())

  def drawToWorld(self, worldMatrix):
    frame = worldMatrix[self.tl[1] : self.tl[1]+self.kernelSize[1], self.tl[0]:self.tl[0]+self.kernelSize[0]]
    frame += self.kernel
    worldMatrix = normalizeMatrix(worldMatrix)



topStack = np.zeros((65, 550))
bottomStack = np.zeros((170,550))


# Define the denoising kernel
denoiseKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))


#for frameNumber, filename in enumerate(os.listdir("../data/videoFrames")):
for frameNumber, notUsefulAtAll in enumerate(os.listdir(FRAMES_PATH)):
  ######## FRAME ACQUISITION
  filename = FILE_STRING%(frameNumber+1)
  print("Processing image " + filename + "...")
  # Load image...
  image = cv2.imread("../data/strafeSquare/" + filename, cv2.IMREAD_COLOR)
  image[0,0] = (0,0,0)
  # Dewarp
  dewarped = image.reshape((image.shape[0] * image.shape[1], 3))[Dewarper.dewarpData['lookup-vector']].reshape((dewarpedDims[1], dewarpedDims[0],3))

  ######## COLOUR SEGMENTATION

  # Convert to HSV from BGR
  dewarpedHSV = cv2.cvtColor(dewarped, cv2.COLOR_RGB2HSV)
  dewarpedHSV = dewarpedHSV[65:,:] # Crop
  # Red threshold
  dewarpedRedThreshed = cv2.inRange(dewarpedHSV, LOWER_RED, UPPER_RED)
  dewarpedRedThreshed = np.vstack((topStack, dewarpedRedThreshed)) # Anti-crop
  # Green threshold
  dewarpedGreenThreshed = cv2.inRange(dewarpedHSV, LOWER_GREEN, UPPER_GREEN)
  dewarpedGreenThreshed = np.vstack((topStack, dewarpedGreenThreshed)) # Anti-crop

  dewarpedMask = cv2.bitwise_or(dewarpedRedThreshed, dewarpedGreenThreshed)
  dewarpedMask = cv2.morphologyEx(dewarpedMask, cv2.MORPH_OPEN, denoiseKernel, iterations=1)
  dewarpedMask = dewarpedMask.astype(np.uint8)

  ######### WORLD ASSEMBLY
  # Find the barrels
  contours = cv2.findContours(dewarpedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1] # I'm not sure why this is needed to be honest.

  barrels = []
  barrelBaseAngles = []
  for c in contours:
    if cv2.contourArea(c) >= MINIMUM_BLOB_SIZE:
      # Actually populate the barrel list
      x,y,w,h = cv2.boundingRect(c)
      b = Barrel(x,y,w, robotOrientation)
      barrels.append(b)
      barrelBaseAngles.append(b.getAngle())
      
      # Render the output for fun ## DRAWING ############
      cv2.drawContours(dewarped, [c], 0, (0,0,255), 1)
      cv2.rectangle(dewarped, (x, y), (x+w, y+h), (0,255,0), 1)
      cv2.putText(dewarped, "%.1f"%(b.distance), (x,y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
      ###################################################

  #TODO: If this is the first frame then do something different. Note that if the above were put into a
  # function (as it should be), then the function should be called, then the frame loop should be called
  # to remove this check from the loop. Failing that, the world should be initialised with a gradient
  # whereby the center has slightly higher mass than the outside to bias first placement toward the center.
  if frameNumber == 0:
    for b in barrels:
      b.drawToKernel(worldMap, robotPosition)
    robotCoordinates.append(robotPosition)
    continue

  if ROTATION:
    # Rotation:
    # Try to position the robot at a number of rotations
    highestLikelihood = 0
    relativeAngle = 0
    highestLikelihoodPosition = robotPosition
    highestLikelihoodCollection = None
    #for a in [math.ceil(((i%2)*2-1)*i/2) for i in range(29)]:
    for a in range(-29,30):
      aRad = a/180*math.pi
      for (b, barrelAngle) in zip(barrels, barrelBaseAngles):
        b.setAngle(barrelAngle + aRad)
      barrelCollection = BarrelCollection(barrels)
      pos, likelihood = barrelCollection.matchToWorld(worldMap, robotPosition)
      vals.append(likelihood)
      angles.append(aRad)
      if(likelihood > highestLikelihood):
        highestLikelihood = likelihood
        highestLikelihoodPosition = pos
        relativeAngle = aRad
        highestLikelihoodCollection = barrelCollection

  else:
    # No rotation:
    highestLikelihoodCollection = BarrelCollection(barrels)
    relativeAngle = 0
    highestLikelihoodPosition, l = highestLikelihoodCollection.matchToWorld(worldMap, robotPosition)


  # Now draw the kernel to the real world in it's correct location (and, implicitly, rotation) to update the world map
  highestLikelihoodCollection.drawToWorld(worldMap)
  robotOrientation += relativeAngle
  robotPosition = highestLikelihoodPosition

  ###### DRAWING DEBUG #######################
  robotCoordinates.append(robotPosition)
  cv2.imwrite("output/detections/"+filename, dewarped); # Save the thresholded image
  worldMapColor = rescaleMatrix(worldMap.copy())
  worldMapColor = np.dstack((worldMapColor, worldMapColor, worldMapColor))
  for i in range(1,len(robotCoordinates)):
    cv2.line(worldMapColor, intTuple(robotCoordinates[i-1]), intTuple(robotCoordinates[i]), (0,0,255), 1)
  viewDirection = intTuple((math.cos(robotOrientation)*ROBOT_VIEW_DIR_LEN, math.sin(robotOrientation)*ROBOT_VIEW_DIR_LEN))
  cv2.line(worldMapColor, intTuple(robotPosition), intTuple((robotPosition[0]+viewDirection[0], robotPosition[1]+viewDirection[1])), (0,255,0), 1)
  cv2.imwrite("output/worldMaps/"+filename, worldMapColor)
  #cv2.imwrite("output/barrelKernels/"+filename, rescaleMatrix(highestLikelihoodCollection.kernel.copy()))
  print("Robot Position: ", robotPosition)
  print("Robot Rotation: ", robotOrientation)


