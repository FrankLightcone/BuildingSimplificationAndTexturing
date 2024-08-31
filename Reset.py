"""
This code could reposition the simplified buildings to their original position.
Author: Ruijie Fan
Date: 2024.8.31
"""

import cv2
import numpy as np
import texturePainted
import pandas as pd
import VectorBased.VectorSimplification as vs


def RotateContourByAngle(img, angle):
    # 查找白色连通域
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    # 计算中心点
    M = cv2.moments(contour)
    if M["m00"] == 0:
        print("Warning: m00 is 0")
        return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # 旋转图片
    rotated_image = cv2.warpAffine(binary, rotation_matrix, (binary.shape[1], binary.shape[0]))
    # 提取旋转后的连通域
    rotated_contours, _ = cv2.findContours(rotated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rotated_contour = rotated_contours[0]
    return rotated_contour


def MoveContourByPosition(contour, position):
    # 计算中心点
    M = cv2.moments(contour)
    if M["m00"] == 0:
        print("Warning: m00 is 0")
        return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # 计算偏移量
    offsetX = position[0] - cX
    offsetY = position[1] - cY
    # 平移
    for point in contour:
        point[0][0] += offsetX
        point[0][1] += offsetY
    return contour


def calculate_area(contour):
    return cv2.contourArea(contour)


def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def calculate_perimeter(contour):
    return cv2.arcLength(contour, True)


def calculate_surface_distance(contour1, contour2, image_shape):
    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)

    cv2.drawContours(mask1, [contour1], -1, 255, -1)
    cv2.drawContours(mask2, [contour2], -1, 255, -1)

    intersection = np.sum((mask1 == 255) & (mask2 == 255))
    union = np.sum((mask1 == 255) | (mask2 == 255))

    if union == 0:
        return None

    surface_distance = 1 - (intersection / union)
    return surface_distance


class Reset:
    def __init__(self, image, ini) -> None:
        self.OriginalImage = image
        self.ColorsINI = ini

    def OriImage(self, image):
        self.OriginalImage = image

    def DrawContourToImage(self, contour):
        cv2.drawContours(self.OriginalImage, [contour], -1, (255, 255, 255), -1)

    def run(self, contour, position, angle, ori_contour):
        contour = RotateContourByAngle(contour, angle)
        if contour is None:
            return -1
        contour = MoveContourByPosition(contour, position)

        if contour is None:
            return -1

        ori_area = calculate_area(ori_contour)
        ori_centroid = calculate_centroid(ori_contour)
        ori_perimeter = calculate_perimeter(ori_contour)

        area = calculate_area(contour)
        centroid = calculate_centroid(contour)
        perimeter = calculate_perimeter(contour)

        surface_distance = calculate_surface_distance(ori_contour, contour, self.OriginalImage.shape)

        # 用传统的矢量方法进行建筑物化简
        vec_data = vs.calculate_metrics_for_contour(ori_contour)
        # 将这些信息追加到一个csv文件末尾
        data = {
            'surface_distance': surface_distance,
            'differ_area': area - ori_area,
            'differ_pos': ((centroid[0] - ori_centroid[0]) ** 2 + (centroid[1] - ori_centroid[1]) ** 2) ** 0.5,
            'differ_perimeter': perimeter - ori_perimeter,

            'surface_distance_vec1': vec_data[0]['surface_distance'],
            'differ_area_vec1': vec_data[0]['differ_area'],
            'differ_pos_vec1': vec_data[0]['differ_pos'],
            'differ_perimeter_vec1': vec_data[0]['differ_perimeter'],

            'surface_distance_vec2': vec_data[1]['surface_distance'],
            'differ_area_vec2': vec_data[1]['differ_area'],
            'differ_pos_vec2': vec_data[1]['differ_pos'],
            'differ_perimeter_vec2': vec_data[1]['differ_perimeter'],

            'surface_distance_vec3': vec_data[2]['surface_distance'],
            'differ_area_vec3': vec_data[2]['differ_area'],
            'differ_pos_vec3': vec_data[2]['differ_pos'],
            'differ_perimeter_vec3': vec_data[2]['differ_perimeter'],

            'surface_distance_vec4': vec_data[3]['surface_distance'],
            'differ_area_vec4': vec_data[3]['differ_area'],
            'differ_pos_vec4': vec_data[3]['differ_pos'],
            'differ_perimeter_vec4': vec_data[3]['differ_perimeter']
        }
        df = pd.DataFrame(data, index=[0])
        df.to_csv('reset.csv', mode='a', header=False, index=False)

        txp = texturePainted.TexturePainted(contour, ori_contour, self.OriginalImage, self.ColorsINI)

        txp.run()
        self.OriginalImage = txp.image

        # TODO：contour 加入TexturePainted,绘制简化后的区域，再将简化前的原图放上去
