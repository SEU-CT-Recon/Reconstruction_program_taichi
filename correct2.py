import numpy as np
from PIL import Image

def repair_bad_pixels(img_array):
    # 读取图像和坏点坐标
   
    bad_pixels_path = 'C:/Users/xji/Desktop/xuchenglong_code_test/source_code/bad_pixels.txt'
    bad_pixels = []
    with open(bad_pixels_path, 'r') as f:
        for line in f:
            y, x = map(int, line.strip().split(','))
            bad_pixels.append((y, x))

    # 创建坏点掩码
    mask = np.zeros_like(img_array, dtype=bool)
    for y, x in bad_pixels:
        mask[y, x] = True

    # 修复函数
    def interpolate_pixel(y, x):
        neighbors = []
        # 搜索左右像素
        for dx in [-1, 1]:
            nx = x + dx
            if 0 <= nx < img_array.shape[1] and not mask[y, nx]:
                neighbors.append((img_array[y, nx], abs(dx)))
        # 搜索上下像素
        for dy in [-1, 1]:
            ny = y + dy
            if 0 <= ny < img_array.shape[0] and not mask[ny, x]:
                neighbors.append((img_array[ny, x], abs(dy)))
        # 对角像素（可选）
        for dy in [-1, 1]:
            for dx in [-1, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < img_array.shape[0] and 0 <= nx < img_array.shape[1] and not mask[ny, nx]:
                    neighbors.append((img_array[ny, nx], np.sqrt(dx**2 + dy**2)))

        if not neighbors:
            return 0.0  # 默认值（可根据需求调整）

        # 加权平均（距离越近权重越大）
        values, distances = zip(*neighbors)
        weights = 1 / np.array(distances)
        return np.sum(values * weights) / np.sum(weights)
    # 修复所有坏点
    repaired_array = img_array.copy()
    for y, x in bad_pixels:
        repaired_array[y, x] = interpolate_pixel(y, x)
    return repaired_array
