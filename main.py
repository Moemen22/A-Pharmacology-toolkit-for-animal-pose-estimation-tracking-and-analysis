import csv

import cv2
import numpy as np
import imutils
import pickle
import pandas as pd 
from csv import DictWriter
from RatDetection import RatDetection
from onedollar import TrajectoryClasification,TrajectoryClasificationStrategy,oneDollorRecognize
from resize import resize
from Meanshift import Mean_Shift
import math
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
trajectory = []
k = 0
q = []
cent = None
quarterFrame=0.0
flag_count1 = 0
flag_count2 = 0
flag_count3 = 0
flag_count4 = 0
flag_count5 = 0
flag_count6 = 0
flag_count7 = 0
flag_count8 = 0
flag_count9 = 0
flag_count10 = 0
flag_count11 = 0
flag_count12 = 0
flag_count13 = 0
flag_count14 = 0
flag_count15 = 0
flag_count16 = 0
count = 0
pp = 0
videoNumber = 1
trajectoryType = []
confidence = []
stop = 0
flagRatIsIn = 1
stopLatancy = 1
checkflag = 0
latancySec = 0
latancyframe = 0.0
i2 = 0
i = 0
pointsList = []
timeFlag = 0
programsec = 0

cap = cv2.VideoCapture('123178.mp4')  # '%i.mp4'%(videoNumber))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

backSub = cv2.createBackgroundSubtractorKNN()

pointtoQuater = ''
frameCounter = 0
ratDetectionMethods = RatDetection()
frameEdit = resize()
trajectoryClassifiction = TrajectoryClasification(oneDollorRecognize())
clf = Mean_Shift()


def preprocess_frame(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 1)

    # Apply thresholding to increase contrast
    ret, thresholded_frame = cv2.threshold(blurred_frame, 120, 255, cv2.THRESH_BINARY)

    return thresholded_frame

def detectRatMask2(img):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    kernel = np.ones((5, 5), np.uint8)
    ret,thresholded_frame = cv2.threshold(imgBlur,120,255,cv2.THRESH_BINARY)
    #fgMask = backSub.apply(thresholded_frame)
    fgMask = backSub.apply(imgBlur)
    mask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 1)
    mask = cv2.dilate(mask, kernel, iterations=0)

    edges = cv2.Canny(mask, 100, 200)
    _, thresh = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return thresh
def detectRatMask(img):
    imgBlur = cv2.GaussianBlur(img, (11, 11), 1)
    kernel = np.ones((7, 7), np.uint8)
    ret, thresholded_frame = cv2.threshold(imgBlur, 150, 255, cv2.THRESH_BINARY)
    fgMask = backSub.apply(imgBlur)
    mask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    edges = cv2.Canny(mask, 100, 200)
    _, thresh = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return thresh


 #def detectRatMask(img):
    #imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
   # kernel = np.ones((5, 5), np.uint8)
   # ret,thresholded_frame = cv2.threshold(imgBlur,120,255,cv2.THRESH_BINARY)
   # #fgMask = backSub.apply(thresholded_frame)
   # fgMask = backSub.apply(imgBlur)
   # mask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
   # mask = cv2.medianBlur(mask, 1)
   # mask = cv2.dilate(mask, kernel, iterations=0)
  #  return mask

def ROI(frame):
    # Y1  Y2   X1   X2
    roi = frame[50: 550, 330: 850]
    return roi


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
pcenter = -1

while (1):
    frameCounter += 1
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break

    ret, frame = cap.read()
    # frame_c=frame
    if ret:
        # frame=frameEdit.rescaleFrame(frame)
        # print(frame.shape)
        # frame=ROI(frame)
        # print(frame.shape)
        # frame_c=frame
        # ratDetectionMethods.sethsvValue(frame)
        # ratDetectionMethods.setframe(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgMask = detectRatMask(frame)
        cnts = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        poly = []
        if ratDetectionMethods.getCountNumberofEnter() < 1:
            latancyframe += 1
        if cap.get(cv2.CAP_PROP_POS_FRAMES) > 20:
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                center = trajectoryClassifiction.determineTheCenter(frame, cnts)
                if pcenter != -1:
                    d = math.sqrt(math.pow(center[0] - pcenter[0], 2) + math.pow(center[1] - center[1], 2))
                    # print (d)
                    if d > 200:
                        continue
                cent = clf.fit(trajectoryClassifiction.getPoints())

                if pp != 3:
                    pp = pp + 1
                if pp == 3:
                    pp = 0

                pcenter = center
              #  trajectory.append(cent[0[0]], cent[0][1])
               # print(trajectory)
                # print (pcenter)
                # print()
                #centroid = Mean_Shift.fit()
                trajectory.append(cent[0])
               # print(trajectory)
                with open('testingggggggggggggg.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['X', 'Y'])  # Writing header row
                    for point in trajectory:
                        writer.writerow(point)

                # Draw red line using the stored trajectory
                thickness = 3
                color = (0, 0, 255)  # BGR color format: red
             #   for i in range(1, len(trajectory)):
              #      cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness)
                #cv2.circle(frame, cent[0], 50, (0, 0, 255), -1)
                #print(cent[0][0])
                #print(cent[0][1])

                #print(cent)
                for cr in trajectoryClassifiction.getPoints():
                    #############################
                    #Mean_Shift.fit(cr)
                    #cv2.circle(frame, centroids, 5, (0, 0, 255), -1)
                    cv2.circle(frame, cent[0], 5, (0, 0, 255), -1)
                    cv2.circle(frame, cr, 5, (0, 0, 255), -1)
                    #################################
                    #####################1/1########################
                    #####################1##########################
                    # 1#1

                    if (cent[0][0] > 400 and cent[0][0] < 630 and cent[0][1] > 230 and cent[0][1] < 460):
                        if (flag_count1 == 0):
                            count = count + 1
                            flag_count1 = 1
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 ==1):
                            flag_count3 =0
                        if(flag_count4 ==1):
                            flag_count4 =0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if(flag_count7 ==1):
                            flag_count7 =0
                        if(flag_count8 ==1):
                            flag_count8 =0
                        if(flag_count9 ==1):
                            flag_count9 =0
                        if(flag_count10 ==1):
                            flag_count10 =0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0

                    # 1#2
                    if (cent[0][0] > 640 and cent[0][0] < 840 and cent[0][1] > 230 and cent[0][1] < 460):
                        if (flag_count2 == 0):
                            count = count + 1
                            flag_count2 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 1#3
                    if (cent[0][0] > 845 and cent[0][0] < 1070 and cent[0][1] > 230 and cent[0][1] < 460):
                        if (flag_count3 == 0):
                            count = count + 1
                            flag_count3 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 1#4
                    if (cent[0][0] > 1075 and cent[0][0] < 1350 and cent[0][1] > 230 and cent[0][1] < 460):
                        if (flag_count4 == 0):
                            count = count + 1
                            flag_count4 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 2#1
                    if (cent[0][0] > 340 and cent[0][0] < 610 and cent[0][1] > 460 and cent[0][1] < 660):
                        if (flag_count5 == 0):
                            count = count + 1
                            flag_count5 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 2#2
                    if (cent[0][0] > 615 and cent[0][0] < 830 and cent[0][1] > 460 and cent[0][1] < 670):
                        if (flag_count6 == 0):
                            count = count + 1
                            flag_count6 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 2#3
                    if (cent[0][0] > 835 and cent[0][0] < 1060 and cent[0][1] > 460 and cent[0][1] < 670):
                        if (flag_count7 == 0):
                            count = count + 1
                            flag_count7 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 2#4
                    if (cent[0][0] > 1065 and cent[0][0] < 1350 and cent[0][1] > 460 and cent[0][1] < 670):
                        if (flag_count8 == 0):
                            count = count + 1
                            flag_count8 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 3#1
                    if (cent[0][0] > 340 and cent[0][0] < 620 and cent[0][1] > 670 and cent[0][1] < 880):
                        if (flag_count9 == 0):
                            count = count + 1
                            flag_count9 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 3#2
                    if (cent[0][0] > 625 and cent[0][0] < 830 and cent[0][1] > 670 and cent[0][1] < 900):
                        if (flag_count10 == 0):
                            count = count + 1
                            flag_count10 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 3#3
                    if (cent[0][0] > 830 and cent[0][0] < 1050 and cent[0][1] > 680 and cent[0][1] < 900):
                        if (flag_count11 == 0):
                            count = count + 1
                            flag_count11 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 3#4
                    if (cent[0][0] > 1060 and cent[0][0] < 1350 and cent[0][1] > 680 and cent[0][1] < 910):
                        if (flag_count12 == 0):
                            count = count + 1
                            flag_count12 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 4#1
                    if (cent[0][0] > 340 and cent[0][0] < 600 and cent[0][1] > 880 and cent[0][1] < 1150):
                        if (flag_count13 == 0):
                            count = count + 1
                            flag_count13 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 4#2
                    if (cent[0][0] > 605 and cent[0][0] < 830 and cent[0][1] > 900 and cent[0][1] < 1150):
                        if (flag_count14 == 0):
                            count = count + 1
                            flag_count14 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 4#3
                    if (cent[0][0] > 835 and cent[0][0] < 1050 and cent[0][1] > 900 and cent[0][1] < 1150):
                        if (flag_count15 == 0):
                            count = count + 1
                            flag_count15 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 4#4
                    if (cent[0][0] > 1055 and cent[0][0] < 1350 and cent[0][1] > 910 and cent[0][1] < 1150):
                        if (flag_count16 == 0):
                            count = count + 1
                            flag_count16 = 1
                        if (flag_count1 == 1):
                            flag_count = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                if (trajectoryClassifiction.numberOfPoints() == 40):
                    Shape, conf = trajectoryClassifiction.trajectoryType()

                    trajectoryClassifiction.restPoints()
                    if (Shape != None):
                        trajectoryType.append(Shape)
                        confidence.append(conf)
                # print(x)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                if(trajectoryClassifiction.numberOfPoints()> 40):
                    trajectoryClassifiction.restPoints()
                    #count = count + 1
            #


        cv2.putText(frame, "Crossing: " + str(count), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(frame, "1$ ==> : " + str(result), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print(count)
        frame = imutils.resize(frame, width=600)


        # show image
        cv2.imshow('frame', frame)
    else:
        break
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# destroys all window
cap.release()
cv2.destroyAllWindows()