from Function import *


class WeakClassifier(object):
    def __init__(self, sample_set_feature, labels, weights):
        """
        :param sample_set_feature: 样本集的特征张量 batch_size*6*8*16*16
        :param labels:
        :param weights:权重矩阵，由Adaboost传回以更新加权误差
        """
        self.pic_num = len(labels)
        self.best_stump = {}
        self.results = None
        self.error_array = np.zeros_like(labels)  # 若预测正确，对应值为0，预测错误为1
        self.error_rate = float('inf')

        self.feature_vector = self.select_features(sample_set_feature, 16)
        self.build_stump(labels, weights)  # 单级决策树

    def select_features(self, sample_set_feature,n,mode=1):
        """
        将16*16的特征图变为1个特征值->每张图有48维德特征向量
        mode0：先pca降维，再随机抽取n个点加和(耗时过长)
        mode1：每张图的特征值为随机抽取的n个点加和
        mode2：每张图的特征值为固定的n个点加和
        :param mode:
        :param sample_set_feature: batch_size*6*8*16*16
        :param n 随机抽取n个点
        :return: feature_vector  batch_size*48
        """
        feature_vector = []
        for img_feature in sample_set_feature:
            img_feature_vector = []
            for channel_feature in img_feature:
                for flat_feature in channel_feature:
                    # if mode == 0:
                    #     # 先pca降维，再随机抽取n个点加和
                    #     img_feature_vector.append(np.sum(pca(flat_feature, 1)))
                    if mode == 1:
                        # 每张图的特征值为随机抽取的n个点加和
                        feature_mask = np.zeros((16, 16))
                        random_dot = np.concatenate((np.random.choice(list(range(16)), n).reshape(n, 1),
                                                     np.random.choice(list(range(16)), n).reshape(n, 1)), axis=1)
                        for i, j in random_dot:
                            feature_mask[i][j] = 1
                        img_feature_vector.append(np.sum(feature_mask * flat_feature))
                    if mode == 2:
                        # 每张图的特征值为随机个点加和
                        feature_mask = np.random.randint(0, 2, (16, 16))
                        img_feature_vector.append(np.sum(feature_mask * flat_feature))
            feature_vector.append(img_feature_vector)
        return np.array(feature_vector)

    def build_stump(self, labels, weights):
        """
        thresh_array为正负样本特征平均值
        分别判断>,<thresh_array时的正确率，将较小的记为弱分类器的错误率
        :param labels:
        :param weights:
        :return:
        """
        thresh_array = np.sum(self.feature_vector, axis=0) / self.pic_num
        for thresh_type in ['lt', 'gt']:
            results = self.classify(thresh_array, thresh_type)
            err_rate = self.weighted_error(results, labels, weights)
            if err_rate < self.error_rate:
                self.best_stump['thresh_type'] = thresh_type
                self.best_stump['threshold'] = thresh_array
                # print(self.best_stump)
                self.error_rate = err_rate
                self.results = results

    def classify(self, thresh_array, thresh_type):
        """
        :param thresh_array: 阈值
        :param thresh_type: "lt"代表小于，”gt“大于
        :return: result_array与label_array维度相同
        """
        results = np.ones(self.pic_num)
        compare_array = np.ones_like(self.feature_vector)
        if thresh_type == 'lt':
            compare_array[self.feature_vector > thresh_array] = -1
            # print(compare_array)
        elif thresh_type == 'gt':
            compare_array[self.feature_vector < thresh_array] = -1
        for img_index, img_result in enumerate(compare_array):
            if img_result.tolist().count(1) < 24:
                results[img_index] = -1
        return results

    def weighted_error(self, results, labels, weights):
        self.error_array = np.zeros_like(labels)
        self.error_array[results != labels] = 1
        return np.dot(self.error_array, weights)


class Adaboost(object):
    def __init__(self, sample_set_feature, label_array):
        self.pic_num = len(sample_set_feature)
        self.weak_classifier = []
        self.weak_classifier_weight = []
        self.weak_classifier_num = 0

        self.final_result = self.build_adaboost(sample_set_feature, label_array)
        self.error_rate = self.get_error_rate(label_array)

    def build_adaboost(self, sample_set_feature, label_array):
        """
        1.初始化权重矩阵
        2.生成弱分类器，弱分类器传回误差
        3.根据误差计算弱分类器权重
        4.计算样本的权值分布
        4.按弱分类器权重组合各个弱分类器
        :return:
        """
        weights = np.full(self.pic_num, 1 / self.pic_num)
        while self.weak_classifier_num < 20:
            weak_classifier = WeakClassifier(sample_set_feature, label_array, weights)
            error_rate = weak_classifier.error_rate
            # print(error_rate)
            if error_rate >= 0.5:
                continue

            alpha = 0.5 * np.log((1 - error_rate) / error_rate)
            self.weak_classifier.append(weak_classifier.results)
            self.weak_classifier_weight.append(alpha)
            self.weak_classifier_num = len(self.weak_classifier_weight)
            # print(self.weak_classifier_num)

            # error_array中，0为预测正确，减少权重；1为预测错误，增加权重
            error_array = weak_classifier.error_array
            for pic_index in range(self.pic_num):
                if error_array[pic_index] == 1:
                    weights[pic_index] = weights[pic_index] / 2 / error_rate
                elif error_array[pic_index] == 0:
                    weights[pic_index] = weights[pic_index] / 2 / (1 - error_rate)

        final_result = sum(self.weak_classifier[index] * self.weak_classifier_weight[index] for index in
                           range(self.weak_classifier_num))
        return final_result

    def get_error_rate(self, label_array):
        err_array = np.ones(self.pic_num)
        err_array[np.sign(self.final_result) == label_array] = 0
        return sum(err_array) / self.pic_num


if __name__ == "__main__":
    sample_features_6 = np.load(file="sample_6.npy")
    sample_labels_6 = np.load(file="label_6.npy")

    # picture_num = sample_features_6.shape[0]
    # D = np.full(picture_num, 1 / picture_num)
    # stump = WeakClassifier(sample_features_6, sample_labels_6, D)
    # stump_error = stump.error_rate

    adaboost_6 = Adaboost(sample_features_6, sample_labels_6)
    results = adaboost_6.final_result


