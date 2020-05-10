import numpy as np
import joblib
from PIL import Image
from Basic.Function import soft_max, gaussian_blur
from Preprocess import Canny, Gabor, Haar
from Model.Adaboost import Adaboost


class Detector(object):
    def __init__(self, file_path):
        """
            创建数个AdaBoost模型并读取参数
        """
        self.adaboost = [joblib.load(file_path + "ada_model%d.m" % set_index)
                         for set_index in range(10)]

    def preprocess(self, image):
        """
            图像预处理
        :param image        ->image_hsv  1*128*128  image_grey 128*128
        :return feature:    6*8*16*16
        """
        image_hsv = gaussian_blur(np.sum(np.array(image.convert('HSV')), axis=2))[np.newaxis, :, :]
        image_gray = gaussian_blur(np.array(image.convert('L')))
        canny_gray = Canny.canny(image_gray)
        gabor_gray = Gabor.gabor(image_gray)
        feature_map = np.concatenate((canny_gray, gabor_gray, image_hsv), axis=0)
        feature = Haar.haar(feature_map)
        # print(feature.shape)
        return feature

    def predicate(self, feature):
        """
            根据特征送进不同的分类器进行预测
        :param feature: 6*8*16*16
        :return pred_value_list: 各个分类器的结果
        """
        pred_value_list = [self.adaboost[i].pred(feature) for i in range(10)]
        # print(pred_value_list)
        return pred_value_list

    def __call__(self, image):
        feature = self.preprocess(image)
        pred_value_list = self.predicate(feature)
        soft_max_result = soft_max(pred_value_list).tolist()
        return soft_max_result.index(max(soft_max_result))


if __name__ == "__main__":
    img = Image.open('../Parameters/Dataset/test.jpg').resize((128, 128))
    detector = Detector("../Parameters/Model/")
    print(detector(img))
