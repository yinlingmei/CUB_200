from PIL import Image
from os import listdir
import numpy as np
from torch.utils.data import Dataset, DataLoader


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


if __name__ == "__main__":
    dataset = CubDataset("../Cub_10", 128)
    print(dataset[100][0].shape, dataset[100][1])
