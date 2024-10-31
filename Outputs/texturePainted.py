"""
该文件定义了一个类TexturePainted，用于将纹理应用到图像上
This code defines a class TexturePainted, which is used to apply texture to an image.
Author: @Ruijie Fan
Date: 2024.8.31
"""

import cv2
import numpy as np
import random
import ChangeColor
import PaperFigMaker.SingleImageFeatureExtractor as SIFE
import pandas as pd


def save_texture(image, contour):
    # 创建一个与图像同样大小的黑色掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 在掩码上绘制轮廓
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 使用掩码提取轮廓内的纹理
    texture = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow('texture', texture)
    # cv2.waitKey()
    return texture, mask


def back_up(image, texture, mask):
    # 仅将纹理区域拷贝到原图像上
    # result = image.copy()
    cv2.copyTo(texture, mask, image)
    # return result


def generate_number(n):
    num = random.randint(-n, n)
    if num != 0:
        probability = 1 / abs(num)
        if random.random() < probability:
            return num
        else:
            return generate_number(n)
    else:
        return generate_number(n)


def extract_RGB(text):
    return text.split(' ')[0], text.split(' ')[1], text.split(' ')[2]


class TexturePainted:
    def __init__(self, ch_contour, contour, image, ini):
        self.contour = contour
        self.image = image
        self.ch_contour = ch_contour
        self.average_color = None
        self.position = None
        self.ini = ini
        self.feature = None
        self.mask = None

    def CalcuMeanColor(self):
        # 获取连通域的边界框
        x, y, w, h = cv2.boundingRect(self.contour)
        cx, cy, cw, ch = cv2.boundingRect(self.ch_contour)
        self.position = [x, y, w, h]

        # 创建一个全黑的遮罩
        mask = np.zeros_like(self.image)

        # 在遮罩上画出你选择的连通域（白色）
        cv2.drawContours(mask, [self.contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        # cv2.imshow('mask', mask)
        # 使用遮罩获取连通域内的像素
        # region = cv2.bitwise_and(self.image, mask)
        # cv2.imshow('region', region)
        # 计算这些像素的平均颜色
        self.average_color = cv2.mean(self.image, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
        # print("Average Color:", self.average_color)
        # 创建一个新的图像，所有像素都设置为平均颜色
        average_color_image = np.zeros((ch, cw, 3), dtype=np.uint8)
        average_color_image[:] = self.average_color[:3]

        self.mask = mask

        return average_color_image

    def AddNoise(self, average_color_image):
        # 获取图像的形状和像素总数
        height, width, _ = average_color_image.shape
        total_pixels = height * width

        # 计算要更改的像素数量
        num_pixels_to_change = int(total_pixels * 0.6)

        # 对每个要更改的像素
        for _ in range(num_pixels_to_change):
            # 随机选择一个像素
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            # 对每个颜色通道
            for c in range(3):
                # 随机增加或减少2
                change = generate_number(5)
                new_value = average_color_image[y, x, c] + change

                # 确保新值在0到255之间
                new_value = min(max(new_value, 0), 255)

                # 更新像素值
                average_color_image[y, x, c] = new_value

        return average_color_image

    def ApplyTextureToOriginal(self, noisy_image):
        x, y, w, h = cv2.boundingRect(self.ch_contour)
        t, m = save_texture(self.image, self.contour)

        # 创建一个与外接矩形相同大小的遮罩
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [self.ch_contour], -1, 255, thickness=cv2.FILLED, offset=(-x, -y))

        # 将噪声图像应用到原图像上
        region = self.image[y:y + h, x:x + w]
        noisy_image = noisy_image[:region.shape[0], :region.shape[1]]
        mask = mask[:region.shape[0], :region.shape[1]]
        # print(region.shape, noisy_image.shape, mask.shape)
        np.copyto(region, noisy_image, where=(mask[:, :, None] == 255))

        # 将合并后的区域放回原图像
        self.image[y:y + h, x:x + w] = region

    def run(self):
        average_color_image = self.CalcuMeanColor()[0][0]
        BestMatchedImage = self.MatchTexture(average_color_image)

        self.ApplyTextureToOriginal(BestMatchedImage)

    def MatchTexture(self, average_color_image):
        # 生成一张白色的800*600的空白图片
        BestMatchedImage = np.ones((800, 600, 3), np.uint8) * 255
        return BestMatchedImage

        # 计算特征
        # 选用纹理库中的第6张图片
        BestMatchedImage = (
            ChangeColor.change_texture_to_target_color(
                "Texture Library/6.png", average_color_image))
        return BestMatchedImage

        # 使用特征提取器提取特征，得到纹理
        self.feature = SIFE.extract_features_from_masked_region(self.image, cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY))
        if len(self.feature) != 30:
            BestMatchedImage = (
                ChangeColor.change_texture_to_target_color(
                    "Texture Library/6.png", average_color_image))
            return BestMatchedImage
            # 读入纹理库特征值csv文件
        df = pd.read_csv("Texture Library/features.csv")
        # 计算得到最相似的纹理，计算方法为欧氏距离，只取前30个特征，第某行对应的纹理就是x.png
        distance = np.sqrt(np.sum((df.iloc[:, 1:31] - self.feature) ** 2, axis=1))
        min_dis = np.where(distance == np.min(distance))
        index = (min_dis[0] + 1)[0]
        # 读取最相似的纹理
        BestMatchedImagePath = f"Texture Library/{index}.png"
        BestMatchedImage = (
            ChangeColor.change_texture_to_target_color(BestMatchedImagePath, average_color_image)
        )
        return BestMatchedImage
