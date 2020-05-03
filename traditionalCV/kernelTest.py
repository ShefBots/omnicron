import cv2
import math
import numpy as np
import scipy.signal as sig
import time

currentTimeMillis = lambda: int(round(time.time() * 1000))
RADIUS = 15

kernel = np.zeros((100,100))
worldMap = np.zeros((1000,1000))

cv2.circle(kernel, (RADIUS,RADIUS), RADIUS, 1, -1)
cv2.imwrite("test/kernel.png", kernel*255)

cv2.circle(worldMap, (500,500), RADIUS, 1, -1)

#conv = cv2.matchTemplate(worldMap, kernel, 0)
#t = currentTimeMillis()
#validConv = sig.convolve2d(worldMap, kernel, mode='valid')
#print("scipy: " + str(currentTimeMillis()-t) + "ms")
#cv2.imwrite("test/conv-valid.png", validConv)

t = currentTimeMillis()
conv = cv2.filter2D(worldMap, cv2.CV_32F, kernel, borderType=cv2.BORDER_ISOLATED)
print("cv2: " + str(currentTimeMillis()-t) + "ms")
pos = np.argmax(conv)
coords = (pos%conv.shape[0], math.ceil(pos/conv.shape[0]))
print("(" + str(coords[0]) + ", " + str(coords[1]) + ")")
cv2.circle(conv, coords, 3, 128, -1)
cv2.imwrite("test/conv.png", conv)

worldCoords = coords
cv2.circle(worldMap, worldCoords, 3, 0.5, -1)
cv2.rectangle(worldMap, (int(worldCoords[0]-kernel.shape[0]/2), int(worldCoords[1]-kernel.shape[1]/2)), (int(worldCoords[0]+kernel.shape[0]/2), int(worldCoords[1]+kernel.shape[1]/2)), 0.5, 1)
cv2.imwrite("test/worldMap.png", worldMap*255)
#### Wow, OpenCV is so much faster.
