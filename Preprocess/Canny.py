from Basic.Function import convolution
import numpy as np


def canny(image_gray):
    """'
        输入一张灰度图128*128
        1.sobel算子计算梯度
        2.NMS留下局部梯度最大值
        3.八联通将边缘连接起来
    :param image_gray:
    :return:数据处理成3维数据 1*128*128
    """
    grad_x, grad_y = sobel_filter(image_gray)
    nms_gray = nms(grad_x, grad_y)
    canny_gray = eight_connect(nms_gray)
    return canny_gray[np.newaxis, :, :]  # 1*128*128


def sobel_filter(image_gray):
    kernel_gx = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    kernel_gy = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    return convolution(image_gray, kernel_gx), convolution(image_gray, kernel_gy)


def nms(grad_x, grad_y):
    """
        NMS留下梯度方向的局部最大值
    """
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    w, h = grad.shape
    grad_mask = np.zeros((w, h))
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if grad[i, j] == 0:
                grad_mask[i, j] = 0
            else:
                gx = grad_x[i, j]
                gy = grad_y[i, j]
                if abs(gy) > abs(gx):
                    weight = abs(gx / gy)
                    g2 = grad[i, j + 1]
                    g4 = grad[i, j - 1]
                    if gy * gx >= 0:
                        g1 = grad[i + 1, j + 1]
                        g3 = grad[i - 1, j - 1]
                    else:
                        g1 = grad[i - 1, j + 1]
                        g3 = grad[i + 1, j - 1]
                else:
                    weight = abs(gy / gx)
                    g2 = grad[i + 1, j]
                    g4 = grad[i - 1, j]
                    if gy * gx >= 0:
                        g1 = grad[i + 1, j + 1]
                        g3 = grad[i - 1, j - 1]
                    else:
                        g1 = grad[i + 1, j - 1]
                        g3 = grad[i - 1, j + 1]
                temp1 = weight * g1 + (1 - weight) * g2
                temp2 = weight * g3 + (1 - weight) * g4
                if grad[i, j] >= temp1 and grad[i, j] >= temp2:
                    grad_mask[i, j] = 1
    return grad_mask * grad


def eight_connect(nms_grad):
    """
        软性八联通算法
         根据情况决定输出值
         1.5*grad： 超过高阈值 + 连通
         1.2*grad： 超过高阈值 + 不连通
         1.0*grad： 超过低阈值 + 联通
         0.0：      超过低阈值 + 不连通
         0.0：      未超过阈值
    :param nms_grad: 128*128
    :return: feature 边缘特征 1*128*128
    """
    th = 0.3 * np.max(nms_grad)
    tl = 0.2 * np.max(nms_grad)
    w, h = nms_grad.shape
    feature = np.zeros_like(nms_grad)
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if nms_grad[i, j] >= th:
                if (nms_grad[i - 1:i + 2, j - 1:j + 2] >= th).any():
                    feature[i, j] = 1.5 * nms_grad[i, j]
                else:
                    feature[i, j] = 1.2 * nms_grad[i, j]
            elif nms_grad[i, j] >= tl:
                if (nms_grad[i - 1:i + 2, j - 1:j + 2] >= th).any():
                    feature[i, j] = nms_grad[i, j]
    return feature


def batch_canny(batch_gray):
    # 输入 batch_size * 128 * 128 输出 batch_size*1*128*128
    return np.concatenate(([canny(image_gray)[np.newaxis, :, :, :] for image_gray in batch_gray]), axis=0)


if __name__ == "__main__":
    from Basic.Dataset import SingleClass
    from Basic.Function import show_image

    test_set = SingleClass("../cub-10/001.Black_footed_Albatross", 128)
    test_data = test_set(2)
    bc = batch_canny(test_data)  # 36*1*128*128
    show_image(*bc[0:5, 0])
