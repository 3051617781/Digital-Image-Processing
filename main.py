import random
import cv2 as cv
import numpy as np


""" 噪声模型"""
# 椒盐噪声
def sp_noise(image, prob=0.05):
    """
    添加椒盐噪声
    :param image:
    :param prob: 噪声比例
    :return:
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
# 高斯噪声
def gaussian_noise(image, mean=0, var=0.001):
    """
    添加高斯噪声
    :param image:
    :param mean: 均值
    :param var: 方差
    :return:
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out
# 均匀噪声
def un_noise(image, mean=10, sigma=100):
    """
    均匀噪声
    :param image:
    :param mean:
    :param sigma:
    :return:
    """
    a = 2 * mean - np.sqrt(12 * sigma)  # a = -14.64
    b = 2 * mean + np.sqrt(12 * sigma)  # b = 54.64
    noiseUniform = np.random.uniform(a, b, image.shape)
    out = image + noiseUniform
    out = np.uint8(cv.normalize(out, None, 0, 255, cv.NORM_MINMAX))  # 归一化为 [0,255]
    return out

"""滤波模型"""
# 反调和
def harmonic_mean_filter(image, kernel_size = 3):

    # 计算滤波核的大小
    k = kernel_size // 2

    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 创建输出图像
    output_image = np.zeros_like(image)

    # 对每个像素应用反调和均值滤波器
    for i in range(k, height - k):
        for j in range(k, width - k):
            # 提取当前像素周围的区域
            region = image[i - k:i + k + 1, j - k:j + k + 1]
            # 防止除以0
            region = np.where(region == 0, 0.1, region)
            # 计算区域的反调和
            harmonic_mean = np.size(region) / np.sum(1 / region)
            # 将计算好的值赋给输出图像的对应位置
            output_image[i, j] = harmonic_mean

    return output_image

# 最大滤波
def max_filter(image, ksize=3):
    """最大滤波函数"""
    # 获取图像的行数和列数
    rows, cols = image.shape
    # 计算边界宽度
    pad_width = ksize // 2
    # 对图像进行边界扩展
    padded_img = cv.copyMakeBorder(image, pad_width, pad_width, pad_width, pad_width, cv.BORDER_REPLICATE)
    # 创建输出图像
    filtered_img = np.zeros_like(image)

    # 遍历每个像素
    for i in range(rows):
        for j in range(cols):
            # 获取当前像素的邻域
            neighborhood = padded_img[i:i + ksize, j:j + ksize]
            # 计算最大值
            filtered_img[i, j] = np.max(neighborhood)

    return filtered_img

# 最小滤波
def min_filter(image, ksize=3):
    """最小滤波函数"""
    rows, cols = image.shape
    pad_width = ksize // 2
    padded_img = cv.copyMakeBorder(image, pad_width, pad_width, pad_width, pad_width, cv.BORDER_REPLICATE)
    filtered_img = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            neighborhood = padded_img[i:i + ksize, j:j + ksize]
            # 计算最小值
            filtered_img[i, j] = np.min(neighborhood)

    return filtered_img

# 中值滤波
def median_blur(image, ksize = 3):
    return cv.medianBlur(image, ksize)

# 算数平均滤波
def avg_blur(image, tup=(5,5)):
    return cv.blur(image, tup)

if __name__ == '__main__':
    """图像导入"""
    # 导入图像
    file = "zht.png"
    src_path =  "src_pics/" + file
    out_path = "out_pics/" + file.split('.')[0] + '/'

    # 显示原始图片
    img = cv.imread(src_path, flags=0)  # 设置flag=0，读取灰度图
    #
    #
    # # """添加噪声"""
    # 椒盐噪声
    img_sp1 = sp_noise(img, prob=0.02)
    img_sp2 = sp_noise(img, prob=0.05)
    img_sp3 = sp_noise(img, prob=0.1)

    cv.imwrite(out_path + 'img_sp1.jpg', img_sp1)
    cv.imwrite(out_path +'img_sp2.jpg', img_sp2)
    cv.imwrite(out_path +'img_sp3.jpg', img_sp3)
    # #
    # # # 高斯噪声
    img_gaussian1 = gaussian_noise(img, mean=0, var=0.04)
    img_gaussian2 = gaussian_noise(img, mean=0.07, var=0.04)
    img_gaussian3 = gaussian_noise(img, mean=-0.07, var=0.04)

    cv.imwrite(out_path + 'img_gaussian1.jpg', img_gaussian1)
    cv.imwrite(out_path + 'img_gaussian2.jpg', img_gaussian2)
    cv.imwrite(out_path + 'img_gaussian3.jpg', img_gaussian3)
    # #
    # # # 均匀噪声
    img_un1 = un_noise(img, mean=10, sigma=100)
    img_un2 = un_noise(img, mean=100, sigma=500)
    img_un3 = un_noise(img, mean=900, sigma=900)

    cv.imwrite(out_path + 'img_un1.jpg', img_un1)
    cv.imwrite(out_path + 'img_un2.jpg', img_un2)
    cv.imwrite(out_path + 'img_un3.jpg', img_un3)
    #
    # file = "sherioc.jpg"
    # path = 'out_pics/sherioc/' + file
    # img = cv.imread('src_pics/'+ file, 0)
    # # cv.imshow(str(file), img)
    # # outpath = 'out_pics/output/avg_blur/' + file.split('.')[0] + '/'
    # cv.imwrite("a.jpg",img)
    #
    # # """滤波"""
    # # 中值滤波：高斯噪声、均匀噪声、（椒盐good）
    # # medianBlur_img0 = cv.medianBlur(img, ksize=1)
    # # medianBlur_img1 = cv.medianBlur(img, ksize=3)
    # # medianBlur_img2 = cv.medianBlur(img, ksize=5)
    # # medianBlur_img3 = cv.medianBlur(img, ksize=7)
    # # medianBlur_img4 = cv.medianBlur(img, ksize=9)
    # # medianBlur_img5 = cv.medianBlur(img, ksize=11)
    #
    # # cv.imwrite(outpath+'0.jpg' ,medianBlur_img0)
    # # cv.imwrite(outpath+'1.jpg' ,medianBlur_img1)
    # # cv.imwrite(outpath+'2.jpg' ,medianBlur_img2)
    # # cv.imwrite(outpath+'3.jpg' ,medianBlur_img3)
    # # cv.imwrite(outpath+'4.jpg' ,medianBlur_img4)
    # # cv.imwrite(outpath+'5.jpg' ,medianBlur_img5)




    # # 最大
    # max_img2 = max_filter(img, 3)
    # max_img3 = max_filter(img, 5)
    # max_img1 = max_filter(img, 7)
    #
    # # 最小
    # min_img1 = min_filter(img, 3)
    # min_img2 = min_filter(img, 5)
    # min_img3 = min_filter(img, 7)
    #
    # # 反调和
    # hm_img1 = harmonic_mean_filter(img, 3)
    # hm_img2 = harmonic_mean_filter(img, 5)
    # hm_img3 = harmonic_mean_filter(img, 7)


    # 显示图片
    key = cv.waitKey(0)
    cv.destroyAllWindows()
