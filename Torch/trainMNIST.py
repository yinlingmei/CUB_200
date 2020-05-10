import torch
import os
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets

os.environ["CUDA_DEVICE_ORDE R"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class WeightedRandomSampler(Sampler):
    """
    初始化均匀self.weights,输出初次抽样索引列表
    每一轮训练结束后用update函数更新self.weights
    下一轮开始时iter输出抽样索引列表
    """

    def __init__(self, num_samples):
        """
        初始化权重均相同，跟据该权重初次抽样进入网络
        :param num_samples:
        """
        self.weights = torch.zeros(num_samples).fill_(1 / num_samples)
        self.num_samples = num_samples
        self.sampled_index = None

    def __iter__(self):
        self.sampled_index = torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()
        # print(self.weights,self.sampled_index)
        return iter(self.sampled_index)

    def __len__(self):
        return self.num_samples

    def __call__(self, k):
        top_k, top_index = torch.topk(self.weights, k)
        # print(top_index[:100])
        sample_top_k_total = torch.tensor(0)
        for i in range(k):
            sample_top_k_total = torch.where(torch.tensor(self.sampled_index) == top_index[i], sample_top_k_total + 1, sample_top_k_total)
        print(sample_top_k_total.sum())

    def update(self, loss):
        """
        传入loss，计算对应于上轮采样索引的weights，之后更新self.weights，将self.weights对应到[0,1,2……n]的初始样本索引
        :param loss:
        :return:
        """
        self.weights = loss / torch.sum(loss)
        # print(self.weights)
        new_weights = torch.zeros(self.num_samples).fill_(1 / self.num_samples)
        old_sampled_index = self.sampled_index
        for i in range(self.num_samples):
            new_weights[old_sampled_index[i]] = self.weights[i]
        self.weights = new_weights
        # print(self.weights)


class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(32 * 7 * 7, 10)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        # x = self.soft_max(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(229)
    trans = transforms.Compose([transforms.ToTensor(),])  # transforms.Normalize((0.5,), (1.0,))])

    # 60000*1*28*28
    train_set = datasets.MNIST(root='data', train=True, transform=trans, download=True)
    # 10000*1*28*28
    test_set = datasets.MNIST(root='data', train=False, transform=trans)

    targets1 = train_set.targets[train_set.targets == 9][:2000]
    targets2 = train_set.targets[train_set.targets == 4][:3000]
    targets3 = train_set.targets[np.array(train_set.targets != 9) & np.array(train_set.targets != 4)]
    data1 = train_set.data[train_set.targets == 9][:2000]
    data2 = train_set.data[train_set.targets == 4][:3000]
    data3 = train_set.data[np.array(train_set.targets != 9) & np.array(train_set.targets != 4)]
    train_set.targets = torch.cat([targets1, targets2, targets3])
    train_set.data = torch.cat([data1, data2, data3], dim=0)

    batch_size = 10000
    # 初始化sampler
    sampler = WeightedRandomSampler(torch.numel(train_set.targets))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=sampler)# ,sampler=sampler shuffle=True
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    net = ConvolutionNetwork().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(10):
        net.train()
        all_data_loss=[]
        for batch_idx, (image, label) in enumerate(train_loader):
            image_num = torch.numel(label)
            label_one_hot = torch.zeros((image_num, 10))
            for index in range(image_num):
                label_one_hot[index, label[index]] = 1
            pred = net(image.cuda())
            loss = torch.sum(
                -torch.log(nn.Softmax(dim=1)(pred).type(torch.DoubleTensor)) * label_one_hot.type(torch.DoubleTensor),
                dim=1) / batch_size
            all_data_loss.append(loss)
            # loss2 = nn.CrossEntropyLoss()(pred,label.cuda())
            # print(loss.item(),loss2.item(),loss.item()==loss2.item())
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            # if batch_idx % 10 == 0 or batch_idx == len(train_loader):
            # print('epoch:{},batch_index:{},loss:{}'.format(epoch, batch_idx, loss.sum()))
        all_data_loss=torch.cat([loss for loss in all_data_loss])
        sampler.update(all_data_loss)
        sampler(1000)

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = net.forward(images)
            _, predicted = torch.max(outputs.data, 1)  # 取每一行的最大值，返回索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model: {} %'.format(100 * correct / total))
