import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.ndimage.filters import gaussian_filter


def show_image(*picture):
    for pic in picture:
        plt.imshow(pic, cmap='gray')
        plt.show()


def convolution(input_tensor, kernel):
    return convolve(input_tensor, kernel, mode='same')


def data_norm(array):
    return (array - np.mean(array)) / np.std(array)


def gaussian_blur(image, mode=0, kernel_size=5):
    """
        高斯平滑 可以对比scipy的图像操作和卷积操作
    :param image:
    :param mode:
    :param kernel_size:
    :return:
    """
    if mode == 0:
        return gaussian_filter(image, 1)
    else:
        gaussian_kernel = np.zeros([kernel_size, kernel_size])
        center = (kernel_size - 1) / 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                gaussian_kernel[i, j] = np.exp(-1 / 2 * ((i - center) ** 2 + (j - center) ** 2)) / (2 * np.pi)
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        return convolution(image, gaussian_kernel)


def pca(feature_map, n):
    new_data = feature_map - np.mean(feature_map, axis=0)
    cov_mat = np.cov(new_data, rowvar=False)
    # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eig_val, eig_vector = np.linalg.eig(np.mat(cov_mat))
    # 对特征值从小到大排序
    eig_val = np.argsort(eig_val)
    # 最大的n个特征值的下标
    n_eig_val = eig_val[-1:-(n + 1):-1]
    # 最大的n个特征值对应的特征向量
    n_eig_vector = eig_vector[:, n_eig_val]
    return np.dot(new_data, n_eig_vector)


def soft_max(results):
    return np.exp(results) / sum(np.exp(results))


if __name__ == "__main__":
    from Basic.Dataset import SingleClass
    from scipy.ndimage import sobel

    test_set = SingleClass("../cub-10/001.Black_footed_Albatross", 128)
    test_data = test_set(2)[0]
    show_image(test_data)
    # Gauss测试
    test_gaussian = np.array([[0.3678, 0.6065, 0.3678],
                              [0.6065, 1.0000, 0.6065],
                              [0.3678, 0.6065, 0.3678]])
    show_image(convolution(test_data, test_gaussian), gaussian_filter(test_data, 1))

    # Sobel测试
    test_sobel = np.array([[-1 - 1j, 0 - 2j, +1 - 1j],
                           [-2 + 0j, 0 + 0j, +2 + 0j],
                           [-1 + 1j, 0 + 2j, +1 + 1j]])
    c = convolution(test_data, test_sobel)
    show_image(np.real(c), np.imag(c))
    show_image(sobel(test_data))

    feature = np.array(((2.5, 2.4), (0.5, 0.7), (2.2, 2.9), (1.9, 2.2), (3.1, 3.0), (2.3, 2.7), (2, 1.6), (1, 1.1),
                        (1.5, 1.6), (1.1, 0.9)))
    low_feature = pca(feature, 1)
