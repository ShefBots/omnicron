# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

import serial
from serproto2 import *
from signal import signal, SIGINT
from sys import exit

ser = serial.Serial('/dev/ttyACM0', timeout=1)
stopMoving(ser)
resetMotionOrigin(ser)

identify(ser)

s = ser.read(4)        # read up to ten bytes (timeout)
for b in s:
    print(hex(b))

if s[2] != 0:
    ser.close()
    raise Exception('not the right serial device!')

running = True
def handler(signal_received, frame):
    running = False
    stopMoving(ser)
    ser.close()
    exit(0)

signal(SIGINT, handler)

#rotationspeed = 45
#rotationspeed = 60
rotationspeed = 90
#rotationspeed = 120
basespeed = 0.3

xsize = 1024
ysize = 768
#xsize = 800
#ysize = 600
#xsize = 640
#ysize = 480

# trim horizon
#verticalcutoff = 0.4
verticalcutoff = 0.25

# https://www.learnopencv.com/invisibility-cloak-using-color-detection-and-segmentation-with-opencv/
lower_red1 = np.array([0,120,70])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([170,120,70])
upper_red2 = np.array([180,255,255])

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (xsize, ysize)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(xsize, ysize))

# allow the camera to warmup
time.sleep(0.1)


while running == True:

    # scan for red
    print("hunt for the red october^H^H^H^H^H^mine")
    spin = np.array([])
    spintime = time.time()
    setAngularVelocity(ser, rotationspeed)
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        im = frame.array

        im = im[round(ysize*verticalcutoff):,:,:]

        im2 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(im2, lower_red1, upper_red1)
        mask2 = cv2.inRange(im2, lower_red2, upper_red2)
        mask = mask1 + mask2

        now = time.time() - spintime
        spin = np.append(spin, [now, mask.sum()])

        rawCapture.truncate(0)
        if now >= 360/rotationspeed-0.1:
            break
    setAllVelocities(ser, 0, 0, 0)

    time.sleep(0.5)

    # if no red is detected....?


    # turn towards red
    thetime = spin[0::2]
    thered = spin[1::2]
    if thered.sum() < 5000:
        print("no red found")
        continue

    maxpos = np.argmax(thered)
    maxtime = thetime[maxpos]
    atdegrees = (maxtime*rotationspeed)

    print("red found at %i degrees" % atdegrees)

    if atdegrees > 180: # spinning backwards
        setAngularVelocity(ser, -rotationspeed)
        tosleep = (360/rotationspeed - maxtime) + 0.15
        print("plus a bit")
        if tosleep < 0:
            tosleep = 0
        time.sleep(tosleep)
    else:
        setAngularVelocity(ser, rotationspeed)
        tosleep = maxtime + 0.15
        print("minus a bit")
        if tosleep < 0:
            tosleep = 0
        time.sleep(tosleep)
    setAllVelocities(ser, 0, 0, 0)



    # find distance to red

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500 # should be a factor of resolution
    params.maxArea = 73440000
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)


    print("finding distance to travel to red")
    j = 0
    ydists = np.array([])
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        im = frame.array

        im = im[round(ysize*verticalcutoff):,:,:]

        im2 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(im2, lower_red1, upper_red1)
        mask2 = cv2.inRange(im2, lower_red2, upper_red2)
        mask = mask1 + mask2


        keypoints = detector.detect(mask)
        sz = 0
        msz = -1 # biggest one
        for i in range(len(keypoints)):
            if keypoints[i].size > sz:
                sz = keypoints[i].size
                msz = i
        if msz != -1:
            keypoint = [keypoints[msz]]
            pt = keypoints[msz].pt
            ydists = np.append(ydists, [ysize*(1-verticalcutoff) - pt[1]])
        else:
            keypoint = np.array([])


        rawCapture.truncate(0)
        j += 1
        if j >= 4:
            break


    if len(ydists) == 0:
        print("distance not found, restarting")
        continue


    print("%i pixel equivalents?" % (np.mean(ydists)/((1-verticalcutoff)*ysize)    *360))
    dist = (pow(( np.mean(ydists)/((1-verticalcutoff)*ysize)    *360 ),2)/2000.0+15) # *0.99 # factor of "safety"
    dist = dist * 1.25
    traveltime = (dist/100)/basespeed + 0.3 # plus fudge factor
    print("travel %0.1f cm in %0.1f seconds" % (dist, traveltime))
    #print(f"travel {dist:.1f} cm in {traveltime:.1f} seconds")
    setForwardVelocity(ser, basespeed)
    time.sleep(traveltime)
    setAllVelocities(ser, 0, 0, 0)

    # wait for diffusal
    print("diffusing")
    time.sleep(2)
    print("done")







