from os import listdir
import pickle
from PIL import Image
import numpy as np
from Basic.Function import gaussian_blur
from Preprocess.Canny import batch_canny
from Preprocess.Gabor import batch_gabor
from Preprocess.Haar import batch_haar


class SingleClass(object):
    def __init__(self, class_path, resolution):
        """
            从class_path读取，resize，高斯平滑去噪，存入image_hsv,image_gray
            输出image_gray batch_size*128*128
        :param class_path: 类的路径
        :param resolution: 分辨率
        """
        # self.image_rgb = [] rgb数据未使用
        self.image_hsv = []
        self.image_gray = []
        self.image_num = 0

        for image_path in listdir(class_path):
            image_data = Image.open(class_path + '/' + image_path).resize((resolution, resolution))

            # image_hsv数据相加，便于Haar在2d平面提取特征
            image_hsv = gaussian_blur(np.sum(np.array(image_data.convert('HSV')), axis=2))
            image_gray = gaussian_blur(np.array(image_data.convert('L')))
            # print(image_hsv.shape,image_gray.shape)
            # (128, 128) (128, 128)
            self.image_hsv.append(image_hsv)
            self.image_gray.append(image_gray)
            self.image_num += 1

    def __call__(self, mode):
        if mode == 1:
            return np.array(self.image_hsv)
        elif mode == 2:
            return np.array(self.image_gray)


class DataSet(object):
    def __init__(self, dataset_path, resolution=128):
        """
            读取全部数据 归一化处理 得到每张图的特征张量
        :param dataset_path: 数据集路径
        :param resolution: 分辨率
        """
        dataset = [SingleClass(dataset_path + '\\' + class_path, resolution) for class_path in listdir(dataset_path)]
        self.dataset_count = len(dataset)

        dataset_hsv = np.concatenate(([batch_data(1) for batch_data in dataset]), axis=0)
        dataset_gray = np.concatenate(([batch_data(2) for batch_data in dataset]), axis=0)

        # 统一归一化,保留类别之间的差异性
        # dataset_hsv [] 10个对象 每个对象batch_size*128*128
        dataset_hsv = [(batch_data(1) - np.mean(dataset_hsv)) / np.std(dataset_hsv) for batch_data in dataset]
        dataset_gray = [(batch_data(2) - np.mean(dataset_gray)) / np.std(dataset_gray) for batch_data in dataset]

        # 输入预处理每个对象的形状为batch_size*128*128，输出为batch_size*N*128*128
        dataset_canny = [batch_canny(batch_gray) for batch_gray in dataset_gray]
        dataset_gabor = [batch_gabor(batch_gray) for batch_gray in dataset_gray]
        dataset_hsv = [batch_hsv[:, np.newaxis, :, :] for batch_hsv in dataset_hsv]

        # concat在一起 10,batch_size*6*128*128
        # haar操作输出  10,batch_size*6*8*16*16
        self.preprocess_result = []
        for batch_index in range(len(dataset)):
            batch_feature = np.concatenate(
                (dataset_canny[batch_index], dataset_gabor[batch_index], dataset_hsv[batch_index]), axis=1)
            self.preprocess_result.append(batch_haar(batch_feature))

    def get_sample_features(self, pos_index):
        """
            生成1个采样集，负样本从其余9个样本集中抽取
        """
        # single_neg_num为从其余每个样本集中抽取的个数
        pos_num = len(self.preprocess_result[pos_index])
        single_neg_num = pos_num // (self.dataset_count - 1)

        sample_features = self.preprocess_result[pos_index]
        for class_index in range(self.dataset_count):
            if class_index != pos_index:
                index = np.arange(len(self.preprocess_result))
                np.random.shuffle(index)
                neg_num = index[:single_neg_num]
                sample_features = np.concatenate(
                    (sample_features, self.preprocess_result[class_index][neg_num]), axis=0)

        # 采样标签集
        sample_labels = np.array([1 for _ in range(pos_num)] +
                                 [-1 for _ in range(single_neg_num * (self.dataset_count - 1))])

        # 打乱
        shuffle_index = np.arange(len(sample_labels))
        np.random.shuffle(shuffle_index)

        sample_features = sample_features[shuffle_index]
        sample_labels = sample_labels[shuffle_index]
        return sample_features, sample_labels


if __name__ == "__main__":
    data_path = "../cub-10"
    test_dataset = DataSet(data_path)

    features, labels = list(zip(*[test_dataset.get_sample_features(i) for i in range(10)]))
    # 保存特征
    with open("../Parameters/Dataset/features.pkl", "wb") as ff:
        pickle.dump(features, ff)
    with open("../Parameters/Dataset/labels.pkl", "wb") as fl:
        pickle.dump(labels, fl)

