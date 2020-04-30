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

print("Loading dewarp data...")
import Dewarper
print("Dewarp data loaded!")

class Barrel:
  BARREL_HEIGHT = 80
  BARREL_RADIUS = 25
  ROBOT_EYE_HEIGHT = 185.5

  eyeHeightAboveBarrel = ROBOT_EYE_HEIGHT - BARREL_HEIGHT

  def __init__(self, x, y, width):
    self.viewCoords = (x + int(width/2), y)
    self.angle = float(self.viewCoords[0])/Dewarper.dewarpData['lookup-table'].shape[0] * math.pi*2
    self.distance = math.tan(Dewarper.dewarpData['angles'][y]) * Barrel.eyeHeightAboveBarrel
    self.relativeCoords = np.asarray([math.cos(self.angle) * self.distance, math.sin(self.angle) * self.distance])
    self.kernelSpaceCoords = downscale(self.relativeCoords)

  def drawToKernel(self, m, offset):
    cv2.circle(m, (int(offset[0] + self.kernelSpaceCoords[0]), int(offset[1] + self.kernelSpaceCoords[1])), int(downscale(Barrel.BARREL_RADIUS)), 255, -1)

  def __str__(self):
    return("Distance: %.1f"%(self.distance) + "\nAngle: %.1f"%(self.angle))


MINIMUM_BLOB_SIZE = 100
MM_PER_PIXEL = 1

dewarpedDims = Dewarper.dewarpData['lookup-table'].shape
downscale = lambda x: x/MM_PER_PIXEL
upscale = lambda x: x*MM_PER_PIXEL

topStack = np.zeros((65, 550))
bottomStack = np.zeros((170,550))

lowerRed = np.array([90, 200, 0])
upperRed = np.array([130, 255, 255])
lowerGreen = np.array([10, 110, 0])
upperGreen = np.array([100, 255, 255])

# Define the denoising kernel
denoiseKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

## Interestingly, the below seems to detect almost all barrels:
#lowerAll = np.array([0, 200, 0])
#upperAll = np.array([360, 255, 255])

# Iterate over every image in the dewarped folder, do some detection:
for filename in os.listdir("../data/videoFrames")[:20]:
  if not filename.endswith(".png"):
    continue
  print("Processing image " + filename + "...")
  # Load image...
  image = np.asarray(Image.open("../data/videoFrames/" + filename))
  # Dewarp
  dewarped = image.reshape((image.shape[0] * image.shape[1], 3))[Dewarper.dewarpData['lookup-vector']].reshape((dewarpedDims[1], dewarpedDims[0],3))

  #dewarped = cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB)# There's no point in converting it into the correct colourspace for openCV, it just measn we have to do more work (both here and when filtering for red)

  # Convert to HSV from BGR
  #dewarpedHSV = cv2.cvtColor(dewarped, cv2.COLOR_RGB2HSV)# See comment above
  dewarpedHSV = cv2.cvtColor(dewarped, cv2.COLOR_BGR2HSV)
  dewarpedHSV = dewarpedHSV[65:,:] # Crop
  # Red threshold
  dewarpedRedThreshed = cv2.inRange(dewarpedHSV, lowerRed, upperRed)
  dewarpedRedThreshed = np.vstack((topStack, dewarpedRedThreshed)) # Anti-crop
  # Green threshold
  dewarpedGreenThreshed = cv2.inRange(dewarpedHSV, lowerGreen, upperGreen)
  dewarpedGreenThreshed = np.vstack((topStack, dewarpedGreenThreshed)) # Anti-crop

  dewarpedMask = cv2.bitwise_or(dewarpedRedThreshed, dewarpedGreenThreshed)
  dewarpedMask = cv2.morphologyEx(dewarpedMask, cv2.MORPH_OPEN, denoiseKernel, iterations=1)
  dewarpedMask = dewarpedMask.astype(np.uint8)

  # Find the barrels
  contours = cv2.findContours(dewarpedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1] # I'm not sure why this is needed to be honest.

  barrels = []
  minBounds = np.zeros(2, dtype=np.float16) # [x min, y min]
  maxBounds = np.zeros(2, dtype=np.float16) # [x max, y max]
  for c in contours:
    if cv2.contourArea(c) >= MINIMUM_BLOB_SIZE:
      # Actually populate the barrel list
      x,y,w,h = cv2.boundingRect(c)
      b = Barrel(x,y,w)
      barrels.append(b)
      # Track the min and max bounds
      maxBounds = np.maximum(maxBounds, b.kernelSpaceCoords)
      minBounds = np.minimum(minBounds, b.kernelSpaceCoords)
      
      # Render the output for fun
      cv2.drawContours(dewarped, [c], 0, (0,0,255), 1)
      cv2.rectangle(dewarped, (x, y), (x+w, y+h), (0,255,0), 1)
      cv2.putText(dewarped, "%.1f"%(b.distance), (x,y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
  cv2.imwrite("output/detections/"+filename, dewarped); # Save the thresholded image

  # Generate estimated position kernel
  kernelSize = (int(maxBounds[0]-minBounds[0]), int(maxBounds[1]-minBounds[1]))
  kernelOffset = (-minBounds[0], -minBounds[1])
  barrelsKernel = np.zeros((kernelSize[1],kernelSize[0]))
  worldMap = np.zeros((1400, 1400))
  offset = (700,700)
  print(filename + ": Drawing " + str(len(barrels)) + " barrels")
  cv2.rectangle(worldMap, (int(offset[0] + minBounds[0]), int(offset[1] + minBounds[1])), (int(offset[0] + maxBounds[0]), int(offset[1] + maxBounds[1])), 255)
  for b in barrels:
    b.drawToKernel(worldMap, offset)
    b.drawToKernel(barrelsKernel, kernelOffset)
  cv2.imwrite("output/barrelKernels/"+filename.strip(".png")+"-a.png", worldMap)
  cv2.imwrite("output/barrelKernels/"+filename.strip(".png")+"-b.png", barrelsKernel)

