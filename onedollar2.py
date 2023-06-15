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


class OneDollarRecognizer(TrajectoryClasificationStrategy):
    def startTrajectoy(self, points):
        z = []

        template_red = Template("turn right (red)", [ Point(45,218), Point(45,217), Point(50,215), Point(61,210), Point(73,205), Point(79,203), Point(81,201), Point(85,201), Point(88,200), Point(89,200), Point(91,200), Point(93,199), Point(96,197), Point(98,197), Point(101,196), Point(105,193), Point(107,193), Point(109,190), Point(111,188), Point(115,185), Point(119,182), Point(122,179), Point(126,175), Point(129,170), Point(131,165), Point(133,159), Point(136,153), Point(137,148), Point(139,142), Point(141,136), Point(141,131), Point(141,127), Point(141,123), Point(141,120), Point(141,117), Point(141,114), Point(141,112), Point(141,111), Point(141,109), Point(141,108), Point(139,105), Point(137,103), Point(135,101), Point(133,100), Point(130,98), Point(128,97), Point(126,97), Point(125,97), Point(123,98), Point(121,98), Point(118,99), Point(115,101), Point(113,102), Point(111,102), Point(109,104), Point(107,104), Point(106,105), Point(106,105), Point(106,106), Point(106,109), Point(106,117), Point(105,126), Point(105,135), Point(105,141), Point(105,151), Point(107,160), Point(107,167), Point(108,173), Point(109,174), Point(109,176), Point(109,177), Point(110,178), Point(111,180), Point(112,182), Point(113,184), Point(113,186), Point(115,190), Point(117,194), Point(120,199), Point(122,204), Point(124,208), Point(125,210), Point(126,213), Point(128,214), Point(129,215), Point(131,217), Point(133,219), Point(136,221), Point(139,222), Point(142,223), Point(149,225), Point(157,227), Point(165,228), Point(172,228), Point(177,228), Point(178,228), Point(181,228), Point(181,228)])
        template_blue = Template("turn left (blue)", [ Point(209,162), Point(208,162), Point(197,163), Point(181,163), Point(158,161), Point(138,157), Point(122,153), Point(109,149), Point(97,145), Point(93,142), Point(91,142), Point(90,142), Point(90,139), Point(90,126), Point(90,120), Point(90,111), Point(91,106), Point(93,99), Point(94,93), Point(95,89), Point(98,86), Point(101,83), Point(105,81), Point(109,79), Point(114,76), Point(119,73), Point(123,71), Point(126,69), Point(128,69), Point(129,69), Point(131,67), Point(132,67), Point(133,69), Point(135,70), Point(139,74), Point(143,78), Point(145,83), Point(149,90), Point(152,99), Point(153,106), Point(154,114), Point(155,119), Point(155,122), Point(154,125), Point(143,133), Point(133,139), Point(124,144), Point(117,147), Point(113,149), Point(107,151), Point(101,154), Point(93,155), Point(86,157), Point(78,158), Point(74,159), Point(71,160), Point(68,160), Point(66,160), Point(64,159), Point(62,159), Point(61,157), Point(58,155), Point(56,152), Point(54,149), Point(53,148), Point(52,144), Point(50,141), Point(50,137), Point(49,134), Point(48,130), Point(48,125), Point(47,121), Point(46,117), Point(46,115), Point(45,113), Point(45,112)])
        tmpl_1 = Template('rotate',
                          [Point(509, 193), Point(471, 193), Point(449, 171), Point(453, 145), Point(473, 127),
                           Point(484, 144), Point(490, 177), Point(488, 197), Point(483, 212)])
        tmpl_2 = Template("rotate 2", [ Point(51,155), Point(52,155), Point(52,154), Point(63,142), Point(72,132), Point(84,124), Point(93,118), Point(102,112), Point(109,106), Point(114,103), Point(117,102), Point(119,101), Point(122,100), Point(125,100), Point(128,100), Point(133,101), Point(138,102), Point(142,104), Point(146,106), Point(150,110), Point(154,115), Point(159,121), Point(161,128), Point(162,134), Point(164,141), Point(165,146), Point(165,151), Point(165,155), Point(165,159), Point(163,162), Point(160,165), Point(156,169), Point(151,173), Point(146,176), Point(141,178), Point(135,178), Point(130,177), Point(125,176), Point(122,174), Point(119,173), Point(119,172), Point(118,171), Point(118,170), Point(118,165), Point(121,152), Point(129,138), Point(135,128), Point(140,120), Point(144,113), Point(149,106), Point(154,101), Point(158,98), Point(166,94), Point(169,92), Point(172,90), Point(176,89), Point(181,87), Point(185,85), Point(188,84), Point(191,83), Point(192,83), Point(193,83), Point(194,83)])

        tmpl_3 = Template("rotate 3", [ Point(48,132), Point(49,133), Point(49,134), Point(50,136), Point(51,139), Point(53,142), Point(55,146), Point(59,151), Point(64,156), Point(70,162), Point(75,166), Point(80,169), Point(100,175), Point(117,176), Point(130,175), Point(137,173), Point(142,171), Point(147,170), Point(149,168), Point(151,167), Point(153,163), Point(154,159), Point(155,155), Point(156,148), Point(157,141), Point(157,134), Point(157,128), Point(155,125), Point(152,121), Point(149,119), Point(146,117), Point(143,116), Point(140,116), Point(137,117), Point(133,118), Point(130,118), Point(127,120), Point(125,121), Point(122,123), Point(118,127), Point(114,131), Point(110,137), Point(106,143), Point(103,152), Point(103,157), Point(103,165), Point(104,171), Point(106,174), Point(109,178), Point(114,180), Point(120,180), Point(131,180), Point(141,180), Point(153,179), Point(163,178), Point(172,176), Point(178,174), Point(184,172), Point(188,171), Point(190,169), Point(193,168), Point(196,166), Point(199,165), Point(202,163), Point(205,160), Point(206,159), Point(207,157), Point(209,156)])

        tmpl_4 = Template("rotate 4", [ Point(126,237), Point(127,237), Point(128,237), Point(129,236), Point(133,236), Point(136,235), Point(140,234), Point(143,234), Point(145,233), Point(147,233), Point(148,233), Point(149,232), Point(151,230), Point(152,229), Point(155,226), Point(156,224), Point(156,223), Point(156,221), Point(156,219), Point(156,217), Point(155,216), Point(155,215), Point(155,214), Point(154,214), Point(153,214), Point(152,214), Point(150,214), Point(147,216), Point(144,218), Point(143,219), Point(141,222), Point(140,225), Point(140,227), Point(140,229), Point(140,231), Point(140,232), Point(142,235), Point(143,236), Point(144,239), Point(146,240), Point(148,241), Point(150,241), Point(151,241), Point(152,241), Point(154,241), Point(156,241), Point(157,241), Point(159,241), Point(160,241), Point(161,241), Point(162,241), Point(162,240), Point(163,240), Point(164,238), Point(165,238)])
        tmpl_5 = Template("rotate 5", [ Point(212,227), Point(211,227), Point(211,226), Point(211,225), Point(211,222), Point(212,218), Point(215,212), Point(219,206), Point(222,202), Point(226,199), Point(228,197), Point(231,195), Point(232,194), Point(233,194), Point(233,195), Point(235,197), Point(237,201), Point(241,204), Point(244,208), Point(247,212), Point(249,215), Point(251,217), Point(251,218), Point(251,220), Point(251,224), Point(249,228), Point(247,230), Point(246,232), Point(244,233), Point(243,234), Point(242,235), Point(240,235), Point(240,236), Point(239,236), Point(238,236), Point(238,234), Point(237,233), Point(237,231), Point(236,229), Point(235,227), Point(235,225), Point(235,223), Point(235,221), Point(235,219), Point(235,216), Point(235,213), Point(235,209), Point(236,205), Point(237,203), Point(238,200), Point(239,198), Point(241,196), Point(244,195), Point(247,193), Point(251,191), Point(254,189), Point(257,188), Point(258,187), Point(259,186), Point(261,186), Point(262,186), Point(263,186), Point(264,185), Point(265,184), Point(266,184), Point(267,183)])
        tmpl_6 = Template("rotate 6", [ Point(245,78), Point(246,78), Point(247,78), Point(247,77), Point(249,77), Point(251,76), Point(254,74), Point(256,73), Point(259,73), Point(259,72), Point(260,72), Point(261,72), Point(263,71), Point(264,70), Point(265,70), Point(266,70), Point(267,69), Point(268,69), Point(269,68), Point(271,65), Point(273,64), Point(275,61), Point(277,60), Point(278,58), Point(279,57), Point(279,56), Point(279,55), Point(279,54), Point(279,53), Point(279,52), Point(279,51), Point(277,50), Point(276,50), Point(275,49), Point(273,49), Point(272,48), Point(270,48), Point(269,48), Point(268,48), Point(265,49), Point(264,50), Point(263,51), Point(262,52), Point(262,54), Point(262,55), Point(262,57), Point(262,59), Point(263,62), Point(264,64), Point(266,67), Point(267,68), Point(267,69), Point(268,69), Point(268,70), Point(269,71), Point(270,72), Point(272,73), Point(275,75), Point(277,76), Point(278,76), Point(279,77), Point(280,77)])
        crv = Template("curve", [ Point(185,177), Point(185,178), Point(186,181), Point(190,190), Point(192,199), Point(194,204), Point(196,210), Point(197,215), Point(197,219), Point(197,221), Point(197,222), Point(196,222), Point(184,220), Point(170,215), Point(161,211), Point(153,208), Point(148,204), Point(147,204), Point(146,202), Point(144,199), Point(142,192), Point(141,184), Point(139,176), Point(137,168), Point(136,165), Point(135,162), Point(135,160), Point(135,158)])
        recognizer = Recognizer([template_blue,template_red,tmpl_1,tmpl_2,tmpl_3])
        for pt in points:
            x, y = pt
            z.append(Point(x, y))
        # Call 'recognize(...)' to match a list of 'Point' elements to the previously defined templates.
        result = recognizer.recognize(z)
        shape, conf = result
        # if (shape != None):
        #     print("1$ ==> ", result)  # Output: ('X', 0.733770116545184)

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
    def append_to_list(self,Point):
        self.pointsList.append(Point)