from Basic.Function import *


class LinearSVM(object):
    def __init__(self, sample_set_feature, label_list, c, toler):
        self.pic_num = len(label_list)
        self.dataset_feature = sample_set_feature
        self.labels = label_list
        self.alphas = np.zeros(self.pic_num)
        self.c = c
        self.b = 0
        self.smo(toler)
        self.w = self.get_w()
        self.error_rate = self.error_rate()

    def smo(self, toler, max_iter=30):
        """
        step1 计算误差ei,若满足if条件优化alpha, 再随机选择一个与alpha成对优化的alphaj,计算误差ej
        step2 计算alpha上下界
        step3 计算eta
        step4：更新alphaj并修剪
        step5：更新alphai
        step6：更新b1和b2
        step7：根据b1和b2更新b
        :param sample_set_feature:
        :param label:
        :param c: 松弛变量
        :param toler: 容错率
        :param max_iter:最大迭代次数
        :return:
        """
        iter = 0  # 该变量存储的是在没有任何alpha改变时遍历数据集的次数
        while (iter < max_iter):  # 限制循环迭代次数，也就是在数据集上遍历maxIter次，且不再发生任何alpha修改，则循环停止
            alphaPairsChanged = 0
        for i in range(self.pic_num):
            ei = self.get_error_i(i, self.b)
            if self.labels[i] * ei < -toler and self.alphas[i] < self.c or self.labels[i] * ei > toler and \
                    self.alphas[i] > 0:
                j = self.get_j(i)
                ej = self.get_error_i(j, self.b)
                alphai_old = self.alphas[i].copy()
                alphaj_old = self.alphas[j].copy()

                l, h = self.get_border(i, j)

                eta = 2.0 * np.dot(self.dataset_feature[i, :], self.dataset_feature[j, :]) - np.dot(
                    self.dataset_feature[i, :], self.dataset_feature[i, :]) - np.dot(
                    self.dataset_feature[j, :], self.dataset_feature[j, :])
                if eta >= 0: continue

                self.alphas[j] -= self.labels[j] * (ei - ej) / eta
                self.alphas[j] = np.clip(self.alphas[j], l, h)

                self.alphas[i] += self.labels[j] * self.labels[i] * (alphaj_old - self.labels[j])
                self.get_b(i, j, ei, ej, alphai_old, alphaj_old)
                alphaPairsChanged += 1
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0

    def get_b(self, i, j, ei, ej, alphai_old, alphaj_old):
        b1 = self.b - ei - self.labels[i] * (self.alphas[i] - alphai_old) * np.dot(
            self.dataset_feature[i, :], self.dataset_feature[i, :].T) - self.labels[j] * (
                     self.alphas[j] - alphaj_old) * np.dot(self.dataset_feature[i, :],
                                                          self.dataset_feature[j, :])
        b2 = self.b - ej - self.labels[i] * (self.alphas[i] - alphai_old) * np.dot(
            self.dataset_feature[i, :], self.dataset_feature[j, :].T) - self.labels[j] * (
                     self.alphas[j] - alphaj_old) * np.dot(self.dataset_feature[j, :],
                                                          self.dataset_feature[j, :])
        if 0 < self.alphas[i] < self.c:
            self.b = b1
        elif 0 < self.alphas[j] < self.c:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

    def get_j(self, i):
        j = i
        while j == i:
            j = np.random.randint(0, self.pic_num)
        return j

    def get_border(self, i, j):
        if self.labels[i] != self.labels[j]:
            l = max(0, self.alphas[j] - self.alphas[i])
            h = min(self.c, self.c + self.alphas[j] - self.alphas[i])
        else:
            l = max(0, self.alphas[j] + self.alphas[i] - self.c)
            h = min(self.c, self.alphas[j] + self.alphas[i])
        return l, h

    def get_error_i(self, i, b):
        fxi = 0
        for pic_index in range(self.pic_num):
            fxi += self.alphas[pic_index] * self.labels[pic_index] * np.dot(self.dataset_feature[i, :],
                                                                               self.dataset_feature[
                                                                               pic_index, :])
        ei = fxi + b - self.labels[i]
        return ei

    def get_w(self):
        w = np.zeros((self.dataset_feature.shape[1], 1))
        for pic_index in range(self.pic_num):
            if __name__ == '__main__':
                w = self.labels[pic_index] * self.alphas[pic_index] * self.dataset_feature[pic_index, :].T
        return w

    def error_rate(self):
        final_pred_array = np.dot(self.dataset_feature, self.w, ) + self.b
        mask = (self.labels == np.sign(final_pred_array)).tolist()
        # print(self.label_list == np.sign(final_pred_array),type(mask))
        error_num = mask.count(False)
        return error_num / self.pic_num


if __name__ == "__main__":
    from Basic.Dataset import Dataset
    import os

    path = r"D:\Code\cub-200\cub-10"
    dataset = [0 for _ in range(10)]
    for i, file_path in enumerate(os.listdir(path)):
        dataset[i] = Dataset.Dataset(path + '/' + file_path, 0, 128)

    sample_set_0 = Dataset.DataSet(0, dataset)
    feature_vector_0 = sample_set_0.feature_vector
    label_list_0 = sample_set_0.label_list

    svm = LinearSVM(feature_vector_0, label_list_0, 0.6, 0.5)
