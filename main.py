"""
Main code of the project
Paper: "A raster-based method for building simplification considering shape and texture features
base on remote sensing images"
Author: Ruijie Fan
Date: 2024.8.31

The main function could process an image with a parameter which locates the image
If you want to process other images, please modify the path
"""
import Simplifier
import Preprocessor
import Reset

import numpy as np
from rich.progress import Progress
import os
import cv2
import time


def ReadInI(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data


def main(indexes):
    # 读取图像
    # 判断输入路径是否有效
    if not os.path.exists('val\\label\\val' + str(indexes) + '.tif'):
        return
    label = cv2.imread(f'val\\label\\val{indexes}.tif', -1)
    ori_img = cv2.imread(f'VectorBased\\Blank.png', -1)
    # 将图像转换为三通道图像
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGRA2BGR)
    pure_sim = np.zeros_like(ori_img)

    # 定义Simplifier简化器
    s = Simplifier.Simplifier(0.5, 0.7, 180, 1)
    # 读取纹理库颜色信息
    colors = ReadInI('E:/Code/ImageQuilting/Image Quilting/ImageProcessingTest/bin/Release/texture_lib/result/colors'
                     '.ini')
    # 定义Reset重置器
    r = Reset.Reset(ori_img, colors)
    r_pure = Reset.Reset(pure_sim, colors)

    # 定义Preprocessor预处理器
    p = Preprocessor.Preprocessor(label, 350)
    ori_pos = p.run()
    # print("Test")
    index = 0
    for new_i in p.PreprocessLabelImage:
        s.read_image_by_mat(new_i[0])
        # 运行
        res = s.run()

        if not np.all(s.result == 0) and res == 0:
            r.run(s.result, ori_pos[index], -s.angle, new_i[1])
            r_pure.run(s.result, ori_pos[index], -s.angle, new_i[1])
        index += 1

    cv2.imwrite(f'VectorBased\\image\\val{indexes}.tif', r.OriginalImage)


if __name__ == '__main__':
    # 计时开始
    start = time.time()
    # 创建一个 Progress 对象
    progress = Progress()

    # 使用 progress 对象作为上下文管理器
    '''
    with progress:
        # 添加一个任务，并获取其任务 ID
        task = progress.add_task("[cyan]Processing...", total=299)

        for i in range(1, 300):
            main(i)
            progress.update(task, advance=1)  # 更新任务进度
    '''
    for i in range(1, 300):
        main(i)
    end = time.time()
    print("Time:", end - start)
