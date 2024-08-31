"""
This code defines Preprocessor class, it can read origin images and break them up,
so that the buildings will be processed one by one.
Author: Ruijie Fan
Date: 2024.8.31
"""

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, Label_Image, Padding_Num=128):
        self.contours = None
        self.Padding_Num = Padding_Num
        self.PreprocessLabelImage = []
        if len(Label_Image.shape) == 3:
            self.Label_Image = cv2.cvtColor(Label_Image, cv2.COLOR_BGR2GRAY)
        else:
            self.Label_Image = Label_Image

    def ExtractConnectedArea(self):
        # 查找白色连通域
        _, binary = cv2.threshold(self.Label_Image, 127, 255, cv2.THRESH_BINARY)
        self.contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def GetConnectedArea(self, area_id):
        contour = self.contours[area_id]
        # 计算质心
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return contour, cx, cy

    def GenerateNewImage(self, Area_Info):
        x, y, w, h = cv2.boundingRect(Area_Info[0])
        new_image = np.zeros((h + self.Padding_Num, w + self.Padding_Num), np.uint8)
        for point in Area_Info[0]:  # 遍历所有点, 并将其移动到new_image的中心
            point[0][0] += (h + self.Padding_Num) / 2 - Area_Info[1]
            point[0][1] += (w + self.Padding_Num) / 2 - Area_Info[2]
        # 绘制移动后的连通域到目标图片上
        cv2.drawContours(new_image, [Area_Info[0]], -1, (255, 255, 255), -1)
        # 将所有点重新移动回原来的位置
        for point in Area_Info[0]:
            point[0][0] -= (h + self.Padding_Num) / 2 - Area_Info[1]
            point[0][1] -= (w + self.Padding_Num) / 2 - Area_Info[2]
        return new_image

    def AddingPreprocessLabelImage(self, new_image, ori_contour):
        self.PreprocessLabelImage.append([new_image, ori_contour])

    def SetPaddingNum(self, Padding_Num):
        self.Padding_Num = Padding_Num

    def SetRSImage(self, RS_Image):
        self.RS_Image = RS_Image

    def SetLabelImage(self, Label_Image):
        if len(Label_Image.shape) == 3:
            self.Label_Image = cv2.cvtColor(Label_Image, cv2.COLOR_BGR2GRAY)
        else:
            self.Label_Image = Label_Image

    def ResetPreprocessLabelImage(self):
        self.PreprocessLabelImage = []

    def run(self):
        self.ExtractConnectedArea()
        ori_position = []
        for i, contour in enumerate(self.contours):
            Area_Info = self.GetConnectedArea(i)
            ori_position.append([Area_Info[1], Area_Info[2]])
            new_image = self.GenerateNewImage(Area_Info)
            self.AddingPreprocessLabelImage(new_image, contour)
        return ori_position
