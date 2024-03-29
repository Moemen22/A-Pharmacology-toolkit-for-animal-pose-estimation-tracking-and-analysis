from collections import deque
import numpy as np
import cv2
import argparse
import imutils
from onedollar2 import TrajectoryClasification,TrajectoryClasificationStrategy,OneDollarRecognizer
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Capturing video through webcam
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help='20220516_123718DLC_resnet50_openfieldOct30shuffle1_10000_labeled.mp4')
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])
outfile = open('123718.txt', 'w')
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture('20220516_123718DLC_resnet50_openfieldOct30shuffle1_10000_labeled.mp4')

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
# Start a while loop
trace_array =[]
rear_count = 0
flag_rear = 0
# old_switch = 0
# flag_switch = 0
alg_switch = -1

skip_counter_blyl =0
skip_flag_blyl=0
skip_counter =0
skip_flag_red = 0
rear_count_red = 0
flag_rear_red=0
grooming =0
groomframe = 0
redoldx = 0
redoldy = 0
yellowoldx = 0
yellowoldy = 0
blueoldx = 0
blueoldy = 0
bluepoints = not None
yellowpoints = not None
redpoints = not None
count = 0
XB = 0
YB = 0
XR = 0
YR = 0
YY = 0
XY = 0

turn = None
confidence = None
direction = False
foot_count = 70
algorithm_switcher = TrajectoryClasification(OneDollarRecognizer())
while (1):

    _, imageFrame = camera.read()

    #frame skipping
    if _:
        # cv2.imwrite('frame{:d}.jpg'.format(count), imageFrame)
        count += 5  # i.e. at 30 fps, this advances one second
        camera.set(cv2.CAP_PROP_POS_FRAMES, count)
    else:
        camera.release()
        break

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([160,100,20], np.uint8)
    red_upper = np.array([179,255,255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for yellow color and
    # define mask
    yellow_lower = np.array([25, 52, 72], np.uint8)
    yellow_upper = np.array([102, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([86,224,73], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=yellow_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=blue_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            if(None not in [x,y,w,h]):
                redoldx = XR
                redoldy = YR
                XR,YR,WR,HR = cv2.boundingRect(contour)
            else:
                redpoints = None
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

            # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            if (None not in [x, y, w, h]):
                yellowoldx = XY
                yellowoldy = YY
                XY, YY, WY, HY = cv2.boundingRect(contour)
            else:
                yellowpoints = None
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (48,255,255), 2)

            cv2.putText(imageFrame, "yellow Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (48,255,255))

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            if (None not in [x, y, w, h]):
                blueoldx = XB
                blueoldy = YB
                XB, YB, WB, HB = cv2.boundingRect(contour)
            else:
                bluepoints = None

            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)

            cv2.putText(imageFrame, "Blue Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))

#shapes



    # If the blue and yellow points stay in one area for a while
    if(abs(yellowoldx - XY) <=20 and abs(yellowoldy - YY) <=20 and abs(blueoldx - XB) <=20 and abs(blueoldy - YB) <=20):
        groomframe += 1
        if(groomframe==10):
            grooming += 1
    else:
        groomframe = 0


    #CHECK IF ANY OF THE POINTS ARE OUTSIDE THE DETERMINED AREA
    old_red_circle = Point(redoldx,redoldy)
    red_circle = Point(XR, YR)

    old_blue_circle = Point(blueoldx, blueoldy)
    blue_circle = Point(XB, YB)

    old_yellow_circle = Point(yellowoldx, yellowoldy)
    yellow_circle = Point(XY, YY)
    polygon = Polygon([(393, 222), (380, 1101), (1297, 1166), (1326, 206)])
    BTLFpoly = Polygon([(934, 556), (770, 1188), (1362, 1218), (1428, 539)])
    # if(yellow_circle.distance(old_red_circle)<=old_yellow_circle.distance(old_red_circle) and direction is False and old_red_circle is not None):
    #     alg_switch = -1
    #     direction = True
    # else:
    #     alg_switch = 1
    #     direction = True

    if(abs(blue_circle.distance(old_blue_circle))< 50):
        algorithm_switcher.append_to_list((blue_circle.x,blue_circle.y))
        trace_array.append(blue_circle)
    else:
        algorithm_switcher.append_to_list((old_blue_circle.x,old_blue_circle.y))
        trace_array.append(old_blue_circle)


    if(len(trace_array) % foot_count == 0):
        turn, confidence = algorithm_switcher.trajectoryType()
        algorithm_switcher.restPoints()
    # if(None not in [yellowpoints,bluepoints,redpoints]):
    #     if (abs(blue_circle.distance(old_blue_circle)) >= abs(red_circle.distance(blue_circle)) or
    #             abs(yellow_circle.distance(old_yellow_circle)) >= abs(red_circle.distance(yellow_circle))):
    #        flag_switch = 1
    #
    #     if(flag_switch == 0):
    #        if (not polygon.contains(blue_circle) and not polygon.contains(yellow_circle) and polygon.contains(
    #                red_circle)):
    #            flag_rear = 1
    #        elif (polygon.contains(blue_circle) and polygon.contains(yellow_circle) and polygon.contains(
    #                red_circle)):
    #            if (flag_rear == 1):
    #                rear_count += 1
    #                flag_rear = 0
    #     elif(flag_switch == 1):
    #         flag_switch = 0
    #         if (polygon.contains(blue_circle) and polygon.contains(yellow_circle) and not polygon.contains(red_circle)):
    #             flag_rear_red = 1
    #         elif (polygon.contains(blue_circle) and polygon.contains(yellow_circle) and polygon.contains(red_circle)):
    #             if (flag_rear_red == 1):
    #                 rear_count_red += 1
    # and abs(blue_circle.distance(old_blue_circle)) > abs(red_circle.distance(blue_circle))
    # and abs(yellow_circle.distance(old_yellow_circle)) > abs(red_circle.distance(yellow_circle))

    # old_switch = flag_switch
    # if((abs(blue_circle.distance(old_blue_circle)) >= 200 or abs(yellow_circle.distance(old_yellow_circle))>=200)
    #         and abs(yellow_circle.distance(blue_circle))>=100):
    #     skip_flag_blyl = 1
    # if(abs(red_circle.distance(old_red_circle)) >= 200):
    #     skip_flag_red = 1

    if(len(trace_array) % foot_count == 0 and turn is not None):
        alg_switch *= -1


    if(None not in [yellowpoints,bluepoints,redpoints]):
        if(alg_switch == 1):

            if (skip_flag_blyl == 1):
                skip_counter_blyl += 1
                if (skip_counter_blyl > 3):
                    flag_rear = 0
                    skip_flag_blyl = 0
                    skip_counter_blyl = 0
            else:
                if (not polygon.contains(blue_circle) and not polygon.contains(yellow_circle) and polygon.contains(
                        red_circle)):
                    flag_rear = 1
                elif (polygon.contains(blue_circle) and polygon.contains(yellow_circle) and polygon.contains(
                        red_circle)):
                    if (flag_rear == 1):
                        rear_count += 1
                        flag_rear = 0

        elif(alg_switch == -1):
            if (skip_flag_red == 1):
                skip_counter += 1
                if (skip_counter > 3):
                    flag_rear_red = 0
                    skip_flag_red = 0
                    skip_counter = 0
            else:
                if (polygon.contains(blue_circle) and polygon.contains(yellow_circle) and not polygon.contains(
                        red_circle)):
                    flag_rear_red = 1
                elif (polygon.contains(blue_circle) and polygon.contains(yellow_circle) and polygon.contains(
                        red_circle)):
                    if (flag_rear_red == 1):
                        rear_count_red += 1
                        flag_rear_red = 0


    # if(None not in [yellowpoints,bluepoints,redpoints]):
    #     if (polygon.contains(blue_circle) and polygon.contains(yellow_circle) and not polygon.contains(red_circle)):
    #         flag_rear_red = 1
    #     elif (polygon.contains(blue_circle) and polygon.contains(yellow_circle) and polygon.contains(red_circle)):
    #         if (flag_rear_red == 1):
    #             rear_count_red += 1
    #             flag_rear_red = 0

    cv2.putText(imageFrame,"rear: "+ str(rear_count), (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (255, 0, 0))
    # cv2.putText(imageFrame, "groom: " + str(grooming), (30, 90),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1.0, (255, 0, 0))
    cv2.putText(imageFrame, "rearing_RED: " + str(rear_count_red), (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (255, 0, 0))
    cv2.putText(imageFrame, "switch? " + str(turn) + " conf: " + str(confidence), (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (255, 0, 0))
    imageFrame = imutils.resize(imageFrame, width=900)
    cv2.imshow("Multiple Points", imageFrame)
    key = cv2.waitKey(10) & 0xFF

    outfile.write("rearing: " + str(rear_count) + '\n')
    outfile.write("rearing_RED: " + str(rear_count_red) + '\n')
    outfile.write("grooming: " + str(grooming) + '\n')

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):

        outfile.close()
        break