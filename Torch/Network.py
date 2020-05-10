import torch
import torch.nn as nn


class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNeuralNetwork, self).__init__()
        # 128*128
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        # 16*16
        self.max = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(256, 10))
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.max(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return self.soft_max(x)


if __name__ == "__main__":
    from Dataloader import MyDataset, WeightedSampler
    from torch.utils.data import DataLoader

    dataset = MyDataset('../Data', 128)
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    net = ConvolutionNeuralNetwork().cuda()
    net.load_state_dict(torch.load('test.pkl'))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(10):
        weights = []
        for iteration, (image, label) in enumerate(loader):
            image, label = image.cuda(), label
            pred = net(image)
            loss = torch.sum(-torch.log(pred.type(torch.DoubleTensor)) * label, dim=1)

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            weights.append(loss)
        weights = torch.cat([weights[i] for i in range(len(weights))])
        loader = DataLoader(dataset=dataset, batch_size=100, sampler=WeightedSampler(weights))
        print('epoch:', epoch, 'loss:', weights)
