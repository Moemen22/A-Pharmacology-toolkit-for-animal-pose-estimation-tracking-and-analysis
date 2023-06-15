import string
import cv2
from dollarpy import Recognizer, Template, Point
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import euclidean


class TrajectoryClasificationStrategy(ABC):

    @abstractmethod
    def startTrajectoy(self, points):
        pass


class oneDollorRecognize(TrajectoryClasificationStrategy):
    def startTrajectoy(self, points):
        z = []

        tmpl_1 = Template('rotate',
                          [Point(509, 193), Point(471, 193), Point(449, 171), Point(453, 145), Point(473, 127),
                           Point(484, 144), Point(490, 177), Point(488, 197), Point(483, 212)])

        recognizer = Recognizer([tmpl_1])
        for pt in points:
            x, y = pt
            z.append(Point(x, y))
        # Call 'recognize(...)' to match a list of 'Point' elements to the previously defined templates.
        result = recognizer.recognize(z)
        shape, conf = result
        if (shape != None):
            print("1$ ==> ", result)  # Output: ('X', 0.733770116545184)

        return result

class TrajectoryClasification:
    def __init__(self, trajectoryStrategyType: TrajectoryClasificationStrategy):
        self.pointsList = []
        self.confidence = []
        self.trajectoryStrategyType = trajectoryStrategyType

    def trajectoryType(self):
        chossenTypeResuts = self.trajectoryStrategyType.startTrajectoy(self.pointsList)

        return chossenTypeResuts

    def getPoints(self):
        return self.pointsList

    def restPoints(self):
        self.pointsList = []

    def numberOfPoints(self):
        return len(self.pointsList)

    def determineTheCenter(self, frame, cnts):
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, center, 2, (0, 0, 255), -1)
        self.pointsList.append(center)
        # print(center)
        return center