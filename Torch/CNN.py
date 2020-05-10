from PIL import Image
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class CubDataset(Dataset):
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

    def __getitem__(self, item):
        return self.image[item].astype(np.float32), self.label[item]

    def __len__(self):
        return len(self.label)


def get_loader(dataset_path):
    return DataLoader(dataset=CubDataset(dataset_path, 128), batch_size=128, shuffle=True)


class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()
        # 128*128*128*3
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # 128*64*64*16
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 128*32*32*64
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 128*16*16*256 128*1*1*256  128*256 128*10 softmax dim=1默认按行计算
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(256, 10)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.max(x).reshape((x.size(0), -1))
        x = self.fc(x)
        return self.soft_max(x)


if __name__ == '__main__':
    loader = get_loader('../Data')
    network = ConvolutionNetwork().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    for epoch in range(100):
        correct = 0
        for image, label in loader:
            image = image.cuda()
            label = label.cuda()
            output = network(image)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(output, dim=1)
            correct += (pred == label).sum().item()

        print('epoch:', epoch, 'loss:', loss, 'correct:', correct)

    acc_rate = correct / 294
    # torch.save(network.state_dict(),"cnn.pkl")
    # network.load_state_dict(torch.load())
    # network.load_state_dict(torch.load("test1.pkl"))
