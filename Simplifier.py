"""
This code defined Simplifier class, using to simplify the buildings.
Author: Ruijie Fan
Date: 2024.8.31
"""
import cv2
import numpy as np
import math


class Simplifier:
    def __init__(self, c_ratio, s_ratio, s_num, i_num) -> None:
        self.label = None
        self.result = None
        self.moments = None
        self.area = None
        self.thresh = None
        self.type = None
        self.angle = None
        self.C_RATIO = c_ratio
        self.S_RATIO = s_ratio
        self.S_NUM = s_num
        self.I_NUM = i_num

    def read_image_by_path(self, path):
        # 读取图像
        image = cv2.imread(path)
        if len(image.shape) == 3:
            # 彩色图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, self.thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:
            # 灰度图像
            # 二值化图像
            ret, self.thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        self.thresh = cv2.bitwise_not(self.thresh)

    def read_image_by_mat(self, image):
        if len(image.shape) == 3:
            # 彩色图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, self.thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:
            # 灰度图像
            # 二值化图像
            ret, self.thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        self.thresh = cv2.bitwise_not(self.thresh)

    def check_orthogonal(self, max_c, low_quality, high_quality, min_d):
        # 使用Shi-Tomas角点检测算法
        # 所有的角点
        allCorners = cv2.goodFeaturesToTrack(self.thresh, maxCorners=max_c, qualityLevel=low_quality, minDistance=min_d)
        if allCorners is None:
            print("Warning: No Corner Found!")
            return 2
        # rightCorners = cv2.goodFeaturesToTrackWithQuality(gray, 1000, 0.1, )
        # 直角角点
        rightCorners = cv2.goodFeaturesToTrack(self.thresh,
                                               maxCorners=max_c, qualityLevel=high_quality, minDistance=min_d)
        # 转换角点坐标为整数
        allCornersInt = np.intp(allCorners)
        rightCornersInt = np.intp(rightCorners)
        cRatio = rightCornersInt.shape[0] / allCornersInt.shape[0]
        if cRatio <= self.C_RATIO:
            return 0
        else:
            return 1

    def rotate_to_horizontal(self):
        # 创建一个LSD对象
        lsd = cv2.createLineSegmentDetector(0)
        # 执行检测结果
        d_lines = lsd.detect(self.thresh)
        # 根据距离判断
        distance = 0
        poses = []
        # 得到最长距离的两点
        for dline in d_lines[0]:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if d > distance:
                distance = d
                poses = [x0, y0, x1, y1]

        # 判断直线是否垂直
        if poses[0] != poses[2]:
            # 获得直线正切值
            tan = abs((poses[1] - poses[3]) / (poses[0] - poses[2]))
            # 得到角度，装换到角度制
            angle = math.atan(tan) * 180 / math.pi
        else:
            angle = 90
        # print("Angle Size:", angle)
        # 判断两点方位，分为偏右，偏左，方便旋转图片
        # 偏右情况
        # print("Type: " + "Right Angle" if self.type else "Not Right Angle")
        if (poses[0] < poses[2] and poses[1] > poses[3]) or (poses[0] > poses[2] and poses[1] < poses[3]):
            # 获得椭圆的旋转方向并求出旋转矩阵
            M = cv2.getRotationMatrix2D((int(self.thresh.shape[1] / 2), int(self.thresh.shape[0]) / 2), -angle, 1)
            self.angle = -angle
        else:
            # 偏左情况
            M = cv2.getRotationMatrix2D((int(self.thresh.shape[1] / 2), int(self.thresh.shape[0] / 2)), angle, 1)
            self.angle = angle
        # 进行旋转
        rot = cv2.warpAffine(self.thresh, M, (self.thresh.shape[1], self.thresh.shape[0]))
        return rot

    def fitting_ellipse_and_rotate(self):
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 得到目标轮廓
        cnt = contours[0]
        # 使用椭圆来拟合轮廓
        ellipse = cv2.fitEllipse(cnt)
        # 获得椭圆的旋转方向并求出旋转矩阵
        M = cv2.getRotationMatrix2D(ellipse[0], -90 + ellipse[2], 1)
        self.angle = -90 + ellipse[2]
        # 进行旋转
        thresh_ = cv2.warpAffine(self.thresh, M, (self.thresh.shape[1], self.thresh.shape[0]))
        return thresh_

    def create_superpixel_seeds_and_compute_s_ratio(self):
        # Initialize SEEDS superpixel segmented
        seeds = cv2.ximgproc.createSuperpixelSEEDS(self.thresh.shape[1], self.thresh.shape[0], 1, self.S_NUM,
                                                   self.I_NUM, 0, 4, False)

        # No need to iterate as the iteration number is 0
        seeds.iterate(self.thresh, 0)

        # Get superpixel labels
        label_seeds = seeds.getLabels()
        self.label = label_seeds
        num_superpixels = seeds.getNumberOfSuperpixels()

        # draw segmented lines
        mask_seeds = seeds.getLabelContourMask()
        mask_inv_seeds = cv2.bitwise_not(mask_seeds)
        img_seeds = cv2.bitwise_and(self.thresh, self.thresh, mask=mask_inv_seeds)

        color_img = np.zeros((self.thresh.shape[0], self.thresh.shape[1], 1), np.uint8)
        color_img[:] = 155
        result_ = cv2.bitwise_and(color_img, color_img, mask=mask_seeds)
        diff = cv2.absdiff(result_, img_seeds)
        result = cv2.add(img_seeds, result_)

        # cv2.imshow('superpixel:', diff)
        # cv2.waitKey(0)

        # Compute number of building pixels for each superpixel
        building_mask = (self.thresh == 255)
        building_counts_per_superpixel = np.bincount(label_seeds[building_mask].ravel(), minlength=num_superpixels)

        # Compute number of pixels for each superpixel
        superpixel_counts = np.bincount(label_seeds.ravel(), minlength=num_superpixels)

        # Compute ratio of building pixels to total pixels for each superpixel
        s_ratios = building_counts_per_superpixel / superpixel_counts

        # Identify superpixels where ratio is greater than the threshold
        selected_superpixels = np.where(s_ratios > self.S_RATIO)[0]

        # Compute the total area of selected superpixels
        total_selected_area = superpixel_counts[selected_superpixels].sum()

        return [selected_superpixels.tolist(), total_selected_area, label_seeds]

    def adjust_image_area_to_target(self, select_area, current_area, target_area, label_seeds):
        # Create a new image with selected superpixels set to 255
        mask = np.isin(label_seeds, select_area)
        new_img = np.zeros_like(self.thresh)
        new_img[mask] = 255

        # Define a morphological kernel
        kernel = np.ones((2, 2), dtype=np.uint8)

        # Define a threshold for the while loop to avoid infinite loops, say 1000 iterations
        max_iterations = 1000
        iterations = 0

        while iterations < max_iterations:
            area_ratio = current_area / target_area
            if abs(area_ratio - 1) <= 0.01:
                return new_img

            if area_ratio > 1.01:
                new_img = cv2.erode(new_img, kernel)
                current_area = np.sum(new_img > 0)
            else:
                new_img = cv2.dilate(new_img, kernel)
                current_area = np.sum(new_img > 0)

            iterations += 1

        print("Warning: Max iterations reached without converging to the target area.")
        return new_img

        pass

    def create_image_from_labels(self, n):
        mask = np.isin(self.label, n)
        img = np.where(mask, 255, 0).astype(np.uint8)
        return img

    def run(self):
        self.type = self.check_orthogonal(1000, 0.1, 0.5, 10)
        if self.type == 1:
            self.thresh = self.rotate_to_horizontal()
        elif self.type == 0:
            self.thresh = self.fitting_ellipse_and_rotate()
        else:
            self.thresh = self.thresh
            return -1

        after_super = self.create_superpixel_seeds_and_compute_s_ratio()
        Sim = self.create_image_from_labels(after_super[0])
        kernel = np.ones((8, 8), np.uint8)
        # cv2.imshow('Sim', Sim)
        img_dilate = cv2.dilate(Sim, kernel, iterations=1)

        count = 0
        I_MAX = 100
        while count < I_MAX:
            img_dilate = cv2.dilate(img_dilate, kernel, iterations=1)
            if np.all(img_dilate[np.where(Sim > 0)] > 0):   # Check if the Sim image is cover the ori image
                break
            count += 1
        if count >= I_MAX:
            print("Warning! The Image has dilated over the max times")
        # img_dilate = cv2.dilate(Sim, kernel, iterations=10)
        self.result = img_dilate
        # cv2.imshow('img_dilate', img_dilate)
        return 0

    def show(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
