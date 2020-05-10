import numpy as np


class Classifier(object):
    def __init__(self):
        self.feature_mask = np.random.rand(16, 16)
        self.threshold_vector = None
        self.classifier_mode = 0
        self.classifier_weight = 0

    def build_classifier(self, sampled_features, sampled_labels, weights_mask):
        """
            循环随机mask 直到获得一个有一定区分度的弱分类器
        :param sampled_features:    样本集的特征张量 batch_size*6*8*16*16
        :param sampled_labels:      样本集的标签    batch_size*1
        :param weights_mask:        权重矩阵0 1    batch_size*1
        :return weights_mask_new:   传回AdaBoost以更新数据的权重
        """
        # error_rate = 1
        # while error_rate > 0.5:
        self.feature_mask = np.random.rand(16, 16)
        error_rate = self.calculate_classifier_parameter(sampled_features, sampled_labels, weights_mask)
        # print(error_rate)
        self.classifier_weight = np.log((1 - error_rate) / (error_rate + 1e-4))
        weights_mask_new = self.update_weights_mask(error_rate, sampled_features, sampled_labels, weights_mask)
        return weights_mask_new

    def calculate_feature_vector(self, feature):
        """
            将16*16的特征图随机加权后求和 -> 变为1个特征值 -> 每张图有48维特征向量
        :param feature: 6*8*16*16
        :return vector: 48
        """
        feature_vector = []
        for filter_result in feature:
            for haar_result in filter_result:
                # 16*16
                feature_vector.append(np.sum(haar_result * self.feature_mask))
        return np.array(feature_vector)

    def calculate_classifier_parameter(self, sampled_features, sampled_labels, weights_mask):
        """
            计算所有待分类样本的特征向量均值 根据错误率判断正样本和均值的关系
        :param sampled_features:    样本集的特征张量 batch_size*6*8*16*16
        :param sampled_labels:      样本集的标签    batch_size*1
        :param weights_mask:        权重矩阵0 1    batch_size*1 由AdaBoost传回以更新加权误差
        """
        vector = np.zeros([48])
        for sampled_feature, weight_mask in zip(sampled_features, weights_mask):
            if weight_mask == 1:
                vector += self.calculate_feature_vector(sampled_feature)
        self.threshold_vector = vector / np.sum(weights_mask)

        error_rate = 0
        for sampled_feature, sampled_label, weight_mask in zip(sampled_features, sampled_labels, weights_mask):
            classify_result = self.classify(sampled_feature)
            if classify_result * sampled_label < 0:
                # 记录分类错误的数据
                error_rate += weight_mask
        if error_rate > 0.5:
            self.classifier_mode = 1
            error_rate = 0
            for sampled_feature, sampled_label, weight_mask in zip(sampled_features, sampled_labels, weights_mask):
                classify_result = self.classify(sampled_feature)
                if classify_result * sampled_label < 0:
                    # 记录分类错误的数据
                    error_rate += weight_mask
        return error_rate

    def classify(self, feature):
        """
            classifier_mode -> 0代表正样本应超过均值 1则相反
        :param feature: 一张图的特征张量 6*8*16*16
        :return:        -1~1 属于正/负样本的概率
        """
        sampled_feature_vector = self.calculate_feature_vector(feature)
        if self.classifier_mode == 0:
            classify_output = np.where(sampled_feature_vector >= self.threshold_vector, 1, -1)
        else:
            classify_output = np.where(sampled_feature_vector <= self.threshold_vector, 1, -1)
        return np.sum(classify_output) / 48

    def update_weights_mask(self, error_rate, sampled_features, sampled_labels, weights_mask):
        weights_mask_new = np.zeros_like(weights_mask)
        for index in range(len(weights_mask)):
            classify_result = self.classify(sampled_features[index])
            if classify_result * sampled_labels[index] < 0:
                weights_mask_new[index] = weights_mask[index] / (2 * error_rate)
            else:
                weights_mask_new[index] = weights_mask[index] / (2 - 2 * error_rate)
        return weights_mask_new


if __name__ == "__main__":
    import pickle

    with open("../Parameters/Dataset/features.pkl", "rb") as ff:
        features = pickle.load(ff)
    with open("../Parameters/Dataset/labels.pkl", "rb") as fl:
        labels = pickle.load(fl)

    batch_size = len(features[6])
    test_weights_mask = np.full(batch_size, 1)
    test_weights_mask = test_weights_mask / np.sum(test_weights_mask)
    wc_6 = Classifier()
    error_rate6 = wc_6.build_classifier(features[6], labels[6], test_weights_mask)
    print(error_rate6)
