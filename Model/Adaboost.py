import joblib
import numpy as np
from Model.Classifier import Classifier


class Adaboost(object):
    def __init__(self):
        self.weak_classifier = []

    def build_adaboost(self, sampled_feature, sampled_label):
        """
            1.初始化权重矩阵
            2.生成弱分类器，弱分类器属性已包括自身权重和分类结果
            3.计算新的权值分布
            4.按弱分类器权重组合各个弱分类器得到最终结果
        :param sampled_feature: batch_size*6*8*16*16
        :param sampled_label: batch_size
        """
        weights = np.full(len(sampled_label), 1)
        weights = weights / np.sum(weights)
        while len(self.weak_classifier) < 10 and np.sum(weights) != 0:
            weak_classifier = Classifier()
            weights = weak_classifier.build_classifier(sampled_feature, sampled_label, weights)
            # print(weights)
            self.weak_classifier.append(weak_classifier)

    def pred(self, img_feature):
        """
            调用存储的弱分类器对象的classify方法->n个结果->根据权重组合
        :param img_feature: 6*8*16*16
        :return:
        """
        results = sum([weak_classifier.classify(img_feature) * weak_classifier.classifier_weight
                       for weak_classifier in self.weak_classifier])
        return results

    def get_error_rate(self, sampled_feature, sampled_label):
        error = 0
        all_count = 0
        for feature, label in zip(sampled_feature, sampled_label):
            results = self.pred(feature)
            all_count += 1
            if results * label < 0:
                error += 1
        return error / all_count


def train_adaboost(sampled_features, sampled_labels):
    """
        每个Adaboost训练10轮 保存10个类正确率最低的模型
    :param sampled_features: 10个样本集的特征张量
    :param sampled_labels: 10个样本集的标签
    """
    for set_index in range(10):
        ada_model = Adaboost()
        ada_model.build_adaboost(sampled_features[set_index], sampled_labels[set_index])
        print("Error Rate:", ada_model.get_error_rate(sampled_features[set_index], sampled_labels[set_index]))
        joblib.dump(ada_model, "../Parameters/Model/ada_model%d.m" % set_index)


if __name__ == "__main__":
    import pickle

    with open("../Parameters/Dataset/features.pkl", "rb") as ff:
        features = pickle.load(ff)
    with open("../Parameters/Dataset/labels.pkl", "rb") as fl:
        labels = pickle.load(fl)

    train_adaboost(features, labels)
