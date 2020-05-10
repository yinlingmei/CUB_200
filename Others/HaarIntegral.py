import numpy as np


def fc_integral(channel_data):
    """
        求每个点对应的积分
    """
    w, h = channel_data.shape
    integral = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if i == 0 and j == 0:
                integral[i, j] = channel_data[i, j]
            if i == 0 and j != 0:
                integral[i, j] = channel_data[i, j] + integral[i, j - 1]
            if i != 0 and j == 0:
                integral[i, j] = channel_data[i, j] + integral[i - 1, j]
            else:
                integral[i, j] = channel_data[i, j] + integral[i - 1, j] + integral[i, j - 1] - integral[i - 1, j - 1]
    return integral


def ext_haar_feature(integral_image, haar_type, stride):
    """
        1.原始haar模板遍历图片提取特征，利用积分数组的值计算特征值
        2.haar依次变为原来的2倍，提取特征
    :param integral_image:  积分图像
    :param haar_type:       上白下黑（8*8），左白右黑（横 8*8） 白-黑-白（3格 12*8），白-黑-白-黑（4格 8*8）
    :param stride:          步长
    :return:
    """
    w, h = integral_image.shape
    feature_vector = []
    for haar_index in range(len(haar_type)):
        s, t = haar_type[haar_index]  # s，t是每个模板的初始大小
        R = [s, 2 * s]
        C = [t, 2 * t]
        for j in range(len(R)):
            r = R[j]  # 初始框遍历图像后变大为原来的2倍
            c = C[j]
            for y in range(0, w - r, stride):
                for x in range(0, h - c, stride):
                    if haar_index == 0:
                        white = integral_image[y, x] + integral_image[y + r, x + c // 2] - integral_image[y + r, x] - \
                                integral_image[y, x + c // 2]
                        black = integral_image[y, x + c // 2] + integral_image[y + r, x + c] - integral_image[
                            y + r, x + c // 2] - integral_image[y, x + c]
                    elif haar_index == 1:
                        white = integral_image[y, x] + integral_image[y + r // 2, x + c] - integral_image[y, x + c] - \
                                integral_image[y + r // 2, x]
                        black = integral_image[y + r // 2, x] + integral_image[y + r, x + c] - integral_image[
                            y + r // 2, x + c] - integral_image[y + r, x]
                    elif haar_index == 2:
                        white = integral_image[y + r, x + c // 3] + integral_image[y, x] - integral_image[
                            y, x + c // 3] - integral_image[y + r, x] + integral_image[
                                    y + r, x + c] + integral_image[y, x + 2 * c // 3] - integral_image[y, x + c] - \
                                integral_image[y + r, x + 2 * c // 3]
                        black = 2 * integral_image[y + r, x + 2 * c // 3] + integral_image[y, x + c // 3] - \
                                integral_image[y, x + 2 * c // 3] - integral_image[y + r, x + c // 3]
                    elif haar_index == 3:
                        white = integral_image[y + r // 2, x + c // 2] + integral_image[y, x] - integral_image[
                            y + r // 2, x] - integral_image[y, x + c // 2] + integral_image[
                                    y + r, x + c] + integral_image[y + r // 2, x + c // 2] - integral_image[
                                    y + r // 2, x + c] - integral_image[y + r, x + c // 2]
                        black = integral_image[y + r, x + c // 2] + integral_image[y + r // 2, x] - integral_image[
                            y + r // 2, x + c // 2] - integral_image[y + r, x] + \
                                integral_image[y + r // 2, x + c] + integral_image[y, x + c // 2] - integral_image[
                                    y, x + c] - integral_image[y + r // 2, x + c // 2]
                    else:
                        white = 0
                        black = 0
                    feature = white - black
                    feature_vector.append(feature)
    return feature_vector


def haar(feature_map):
    feature_vector = []
    for i in range(feature_map.shape[2]):
        channel_data = feature_map[:, :, i]
        integral_data = fc_integral(channel_data)
        haar_type = [[8, 8], [8, 8], [8, 12], [8, 8]]
        stride = 2
        feature_vector.append(ext_haar_feature(integral_data, haar_type, stride))
    feature_vector = np.array(feature_vector)
    # print(feature_vector.shape)  通道数*26600
    return feature_vector


if __name__ == "__main__":
    from PIL import Image
    from Basic.Function import data_norm

    image_data = Image.open('../Parameters/Dataset/test.jpg').resize((128, 128))
    rgb_data = data_norm(np.array(image_data))
    feature_data = haar(rgb_data)
