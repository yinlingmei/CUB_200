from Basic.Function import convolution
import numpy as np


def haar(image_feature):
    """
        1.生成4种卷积核 small 卷积核尺寸 8
        2.生成4种卷积核 large 卷积核尺寸 16
        3.池化输出结果
    :param image_feature: 6*128*128
    :return haar_result: 6*8*32*32
    """
    haar_result = []
    for feature in image_feature:
        small_result = np.concatenate(([max_pooling(convolution(feature, haar_kernel))[np.newaxis, :, :]
                                        for haar_kernel in build_haar_kernel(8)]), axis=0)
        large_result = np.concatenate(([max_pooling(convolution(feature, haar_kernel))[np.newaxis, :, :]
                                        for haar_kernel in build_haar_kernel(16)]), axis=0)
        result = np.concatenate((small_result, large_result), axis=0)  # 8*32*32
        haar_result.append(result)
    return np.array(haar_result)


def build_haar_kernel(kernel_size):
    """
        haar窗口 -  上白下黑（8*8），左白右黑（横 8*8） 白-黑-白（1-2-1 8*8），白-黑-白-黑（4格 8*8）
        另外4种分别是他们尺寸的两倍
    """
    d = kernel_size // 4
    haar_kernel = np.ones((4, kernel_size, kernel_size))
    haar_kernel[0, 2 * d:, :] = -1
    haar_kernel[1, :, 2 * d:] = -1
    haar_kernel[2, :, d:3 * d] = -1
    haar_kernel[3, 2 * d:, :2 * d] = -1
    haar_kernel[3, 2 * d:, :2 * d] = -1
    return haar_kernel


def max_pooling(feature_map, stride=8):
    w, h = feature_map.shape
    out_w = w // stride
    out_h = h // stride
    out = np.zeros((out_w, out_h))
    for i in range(out_w):
        for j in range(out_h):
            out[i, j] = np.max(feature_map[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride])
    return out


def batch_haar(batch_feature):
    return np.concatenate(([haar(image_feature)[np.newaxis, :, :, :, :] for image_feature in batch_feature]), axis=0)


if __name__ == "__main__":
    from Basic.Dataset import SingleClass
    from Basic.Function import show_image
    from Preprocess.Canny import batch_canny
    from Preprocess.Gabor import batch_gabor

    test_set = SingleClass("../cub-10/001.Black_footed_Albatross", 128)
    test_data = test_set(2)
    test_canny = batch_canny(test_data)
    test_gabor = batch_gabor(test_data)
    bh = batch_haar(test_gabor)
    bc = batch_haar(test_canny)
    show_image(*bc[0:5, 0, 0])
