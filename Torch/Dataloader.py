import torch
import numpy as np
from PIL import Image
from os import listdir
from torch.utils.data import Dataset, Sampler


class MyDataset(Dataset):
    def __init__(self, dataset_path, resolution):
        self.image = []
        self.label = []

        for index, class_path in enumerate(listdir(dataset_path)):
            for image_path in listdir(dataset_path + '/' + class_path):
                image_data = Image.open(dataset_path + '/' + class_path + '/' + image_path)
                image_data = np.array(image_data.resize((resolution, resolution)))
                image_data = np.transpose(image_data, (2, 0, 1))
                self.image.append(image_data)
                self.label.append(index)

        self.image = (self.image - np.mean(self.image)) / np.std(self.image)
        label_one_hot = np.zeros((len(self.label), 10))
        for i in range(len(self.label)):
            label_one_hot[i, self.label[i]] = 1
        self.label = label_one_hot

    def __getitem__(self, item):
        return self.image[item].astype(np.float32), self.label[item]

    def __len__(self):
        return len(self.label)


class WeightedSampler(Sampler):
    def __init__(self, loss):
        self.weights = torch.cumsum(loss / torch.sum(loss), dim=0)

    def __iter__(self):
        """
        根据loss生成样本的权重0~1,并累加
        生成随机数,落在哪个区间就抽后端点的样本
        :param loss:
        :return:
        """
        sampled_index = []
        for _ in range(torch.numel(self.weights)):
            random_num = np.random.uniform(0, 1)
            if random_num < self.weights[0]:
                sampled_index.append(0)
                continue
            for index in range(torch.numel(self.weights)):
                if self.weights[index] <= random_num < self.weights[index + 1]:
                    sampled_index.append(index + 1)
                    continue
        return iter(sampled_index)

    def __len__(self):
        return len(self.weights.size)


if __name__ == '__main__':
    dataset = MyDataset('..\Data', 128)
    print(dataset[100][0].shape, dataset[100][1])

    # loss = np.random.rand(30) * 10
    # print(list(BatchSampler(WeightedSampler(loss),batch_size=10,drop_last=False)))
    # loader = get_loader(dataset, WeightedSampler(loss))
