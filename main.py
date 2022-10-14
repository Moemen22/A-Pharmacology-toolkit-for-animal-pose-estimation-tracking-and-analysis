import cv2  
import numpy as np
import imutils
import pickle
import pandas as pd 
from csv import DictWriter
from RatDetection import RatDetection
from onedollar import TrajectoryClasification,TrajectoryClasificationStrategy,oneDollorRecognize
from resize import resize
import math

k = 0

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
videoNumber = 1
trajectoryType = []
confidence = []
stop = 0
flagRatIsIn = 1
stopLatancy = 1
checkflag = 0
latancySec = 0
latancyframe = 0.0

pointsList = []
timeFlag = 0
programsec = 0

cap = cv2.VideoCapture('115726.mp4')  # '%i.mp4'%(videoNumber))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

backSub = cv2.createBackgroundSubtractorKNN()

pointtoQuater = ''
frameCounter = 0
ratDetectionMethods = RatDetection()
frameEdit = resize()
trajectoryClassifiction = TrajectoryClasification(oneDollorRecognize())


def detectRatMask(img):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 2)
    kernel = np.ones((5, 5), np.uint8)
    fgMask = backSub.apply(imgBlur)
    mask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


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
                    d = math.sqrt(math.pow(center[0] - pcenter[0], 2) + math.pow(center[1] - pcenter[1], 2))
                    # print (d)
                    if d > 200:
                        continue

                pcenter = center
                # print (pcenter)
                # print()
                i = 0
                for cr in trajectoryClassifiction.getPoints():
                    cv2.circle(frame, cr, 5, (0, 0, 255), -1)
                    #####################1/1########################
                    #####################1##########################
                    # 1#1
                    if (center[0] > 400 and center[0] < 630 and center[1] > 230 and center[1] < 460):
                        if (flag_count1 == 0):
                            count = count + 1
                            flag_count1 = 1
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                    # 1#2
                    if (center[0] > 640 and center[0] < 840 and center[1] > 230 and center[1] < 460):
                        if (flag_count1 == 1):
                            flag_count1 = 0
                        if (flag_count2 == 0):
                            count = count + 1
                            flag_count2 = 1
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                    # 1#3
                    if (center[0] > 845 and center[0] < 1070 and center[1] > 230 and center[1] < 460):
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 0):
                            count = count + 1
                            flag_count3 = 1
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                    # 1#4
                    if (center[0] > 1075 and center[0] < 1350 and center[1] > 230 and center[1] < 460):
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 0):
                            count = count + 1
                            flag_count4 = 1
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                    # 2#1
                    if (center[0] > 340 and center[0] < 610 and center[1] > 460 and center[1] < 660):
                        if (flag_count1 == 1):
                            flag_count1 = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count5 == 0):
                            count = count + 1
                            flag_count5 = 1
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                    # 2#2
                    if (center[0] > 615 and center[0] < 830 and center[1] > 460 and center[1] < 670):
                        if (flag_count1 == 1):
                            flag_count1 = 0
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 0):
                            count = count + 1
                            flag_count6 = 1
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                    # 2#3
                    if (center[0] > 835 and center[0] < 1060 and center[1] > 460 and center[1] < 670):
                        if (flag_count2 == 1):
                            flag_count2 = 0
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 0):
                            count = count + 1
                            flag_count7 = 1
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                    # 2#4
                    if (center[0] > 1065 and center[0] < 1350 and center[1] > 460 and center[1] < 670):
                        if (flag_count3 == 1):
                            flag_count3 = 0
                        if (flag_count4 == 1):
                            flag_count4 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 0):
                            count = count + 1
                            flag_count8 = 1
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                    # 3#1
                    if (center[0] > 340 and center[0] < 620 and center[1] > 670 and center[1] < 880):
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count9 == 0):
                            count = count + 1
                            flag_count9 = 1
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                    # 3#2
                    if (center[0] > 625 and center[0] < 830 and center[1] > 670 and center[1] < 900):
                        if (flag_count5 == 1):
                            flag_count5 = 0
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 0):
                            count = count + 1
                            flag_count10 = 1
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                    # 3#3
                    if (center[0] > 830 and center[0] < 1050 and center[1] > 680 and center[1] < 900):
                        if (flag_count6 == 1):
                            flag_count6 = 0
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 0):
                            count = count + 1
                            flag_count11 = 1
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 3#4
                    if (center[0] > 1060 and center[0] < 1350 and center[1] > 680 and center[1] < 910):
                        if (flag_count7 == 1):
                            flag_count7 = 0
                        if (flag_count8 == 1):
                            flag_count8 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 0):
                            count = count + 1
                            flag_count12 = 1
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 4#1
                    if (center[0] > 340 and center[0] < 600 and center[1] > 880 and center[1] < 1150):
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count13 == 0):
                            count = count + 1
                            flag_count13 = 1
                        if (flag_count14 == 1):
                            flag_count14 = 0
                    # 4#2
                    if (center[0] > 605 and center[0] < 830 and center[1] > 900 and center[1] < 1150):
                        if (flag_count9 == 1):
                            flag_count9 = 0
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count13 == 1):
                            flag_count13 = 0
                        if (flag_count14 == 0):
                            count = count + 1
                            flag_count14 = 1
                        if (flag_count15 == 1):
                            flag_count15 = 0
                    # 4#3
                    if (center[0] > 835 and center[0] < 1050 and center[1] > 900 and center[1] < 1150):
                        if (flag_count10 == 1):
                            flag_count10 = 0
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count14 == 1):
                            flag_count14 = 0
                        if (flag_count15 == 0):
                            count = count + 1
                            flag_count15 = 1
                        if (flag_count16 == 1):
                            flag_count16 = 0
                    # 4#4
                    if (center[0] > 1055 and center[0] < 1350 and center[1] > 910 and center[1] < 1150):
                        if (flag_count11 == 1):
                            flag_count11 = 0
                        if (flag_count12 == 1):
                            flag_count12 = 0
                        if (flag_count15 == 1):
                            flag_count15 = 0
                        if (flag_count16 == 0):
                            count = count + 1
                            flag_count16 = 1

                if (trajectoryClassifiction.numberOfPoints() == 70):
                    Shape, conf = trajectoryClassifiction.trajectoryType()

                    trajectoryClassifiction.restPoints()
                    if (Shape != None):
                        trajectoryType.append(Shape)
                        confidence.append(conf)
                # print(x)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            #

        cv2.putText(frame, "count: " + str(count), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)
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