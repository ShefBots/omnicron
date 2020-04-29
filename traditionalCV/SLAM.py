# Robot ball bearing height is 185.5mm

print("Loading libraries...")
#from picamera.array import PiRGBArray
#from picamera import PiCamera
#import time
import cv2
#import sys
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os

print("Loading dewarp data...")
import Dewarper
print("Dewarp data loaded!")

dewarpedDims = Dewarper.dewarpData['lookup-table'].shape

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 200
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

topStack = np.zeros((65, 550))
bottomStack = np.zeros((170,550))

# Iterate over every image in the dewarped folder, do some detection:
for filename in os.listdir("../data/videoFrames"):
  if not filename.endswith(".png"):
    continue
  print("Processing image " + filename + "...")
  # Load image...
  image = np.asarray(Image.open("../data/videoFrames/" + filename))
  # Dewarp
  dewarped = image.reshape((image.shape[0] * image.shape[1], 3))[Dewarper.dewarpData['lookup-vector']].reshape((dewarpedDims[1], dewarpedDims[0],3))

  # Convert to HSV from BGR
  dewarpedHSV = cv2.cvtColor(dewarped, cv2.COLOR_BGR2HSV)
  lowerRed = np.array([90, 200, 0])
  upperRed = np.array([130, 255, 255])
  dewarpedRedThreshed = cv2.inRange(dewarpedHSV, lowerRed, upperRed)
  dewarpedRedThreshed = dewarpedRedThreshed[65:,:]
  #dewarpedRedThreshed = np.vstack((topStack, dewarpedRedThreshed, bottomStack))
  dewarpedRedThreshed = np.vstack((topStack, dewarpedRedThreshed))

  # Detect the dots
  detector = cv2.SimpleBlobDetector_create(params)
  dewarpedRedThreshed = dewarpedRedThreshed.astype('uint8')
  dewarpedRedThreshed = cv2.bitwise_not(dewarpedRedThreshed)
  keypoints = detector.detect(dewarpedRedThreshed)
  dewarpedRedThreshed = cv2.bitwise_not(dewarpedRedThreshed)
  keyImage = cv2.drawKeypoints(dewarpedRedThreshed, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  cv2.imwrite("output/out-"+filename, keyImage); # Save the thresholded image
