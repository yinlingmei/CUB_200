from Basic.Function import convolution
import numpy as np


def gabor(image_gray):
    """
        1.生成4种gabor滤波器
        2.将每一种gabor滤波器与图像进行卷积
        3.数据处理成三维数组  (4,500,375)
    :param image_gray:128*128
    :return:gabor_result 4*128*128
    """
    kernel_list = []
    for filter_size in [3, 7]:
        for theta in [0, np.pi / 2]:
            kernel_list.append(build_gabor_filter(filter_size, theta))
    gabor_result = np.concatenate(([convolution(image_gray, kern)[np.newaxis, :, :] for kern in kernel_list]), axis=0)
    return gabor_result


def build_gabor_filter(filter_size, theta, sigma=1.12, lamba=2, psi=0, gamma=0.5):
    """
        生成核
    :param sigma: 带宽通常为1，在此情况下，sigma=0.56lamda
    :param theta: 旋转角度 0  90
    :param lamba: 波长
    :param psi: 相位，0和180度分别对应中心对称的center-on函数和center-off函数
    :param gamma:核函数的椭圆率,一般取0.5
    :param filter_size:卷积核大小
    :return kernel: filter_size*filter_size
    """
    kernel_size = (filter_size - 1) / 2
    (x, y) = np.meshgrid(np.arange(-kernel_size, kernel_size + 1), np.arange(-kernel_size, kernel_size + 1))
    # Rotation
    rot_x = x * np.cos(theta) + y * np.sin(theta)
    rot_y = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-0.5 * sigma ** 2 * (rot_x ** 2 + rot_y ** 2 / gamma ** 2)) * np.cos(6.28 / lamba * rot_x + psi)
    return kernel


def batch_gabor(batch_gray):
    return np.concatenate(([gabor(image_gray)[np.newaxis, :, :, :] for image_gray in batch_gray]), axis=0)


# 特征图生成并显示
if __name__ == "__main__":
    from Basic.Dataset import SingleClass
    from Basic.Function import show_image

    test_set = SingleClass("../cub-10/001.Black_footed_Albatross", 128)
    test_data = test_set(2)
    bc = batch_gabor(test_data)  # 36*4*128*128
    show_image(*bc[0:5, 0])
