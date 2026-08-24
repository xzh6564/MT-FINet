import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class GastricDataset(Dataset):
    def __init__(self, csv_file, img_dir1, img_dir2, transform=None):
        self.df = pd.read_csv(csv_file, encoding='gb18030')
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['num'])

        path1 = os.path.join(self.img_dir1, img_name + '.jpg')
        path2 = os.path.join(self.img_dir2, img_name + '.png')

        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        image = torch.cat([img1, img2], dim=0)

        clinical = torch.tensor(
            row[['gender', 'age', 'tumor_size', 'ETE']].values.astype(float),
            dtype=torch.float32)

        cclnm = torch.tensor(row['Label1'], dtype=torch.float32)
        lclnm = torch.tensor(row['Label2'], dtype=torch.float32)

        return image, clinical, cclnm, lclnm


def data_list(datapath: str):
    train_images_path = []
    val_images_path = []
    img_path = [os.path.join(datapath, 'train'), os.path.join(datapath, 'valid')]
    for path in img_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                if path.endswith('/train'):
                    train_images_path.append(os.path.join(root, file))
                elif path.endswith('/valid'):
                    val_images_path.append(os.path.join(root, file))
    return train_images_path, val_images_path


data_path = './data/origin'
label_path1 = './data/train.csv'
label_path2 = './data/valid.csv'
img_dir1 = './data/origin/train'
img_dir2 = './data/origin/valid'
img_dir3 = './data/fat/train'
img_dir4 = './data/fat/valid'


other_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_dataset = GastricDataset(csv_file=label_path1, img_dir1=img_dir1, img_dir2=img_dir3, transform=other_transform)
valid_dataset = GastricDataset(csv_file=label_path2, img_dir1=img_dir2, img_dir2=img_dir4, transform=other_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=min(len(train_dataset), 64), shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=min(len(valid_dataset), 64), shuffle=True)


train_batch = next(iter(train_loader))[0]
valid_batch = next(iter(valid_loader))[0]


train_mean = np.mean(train_batch.numpy(), axis=(0, 2, 3))
train_std = np.std(train_batch.numpy(), axis=(0, 2, 3))

valid_mean = np.mean(valid_batch.numpy(), axis=(0, 2, 3))
valid_std = np.std(valid_batch.numpy(), axis=(0, 2, 3))

print("train_mean:", train_mean)
print("train_std:", train_std)

print("valid_mean:", valid_mean)
print("valid_std:", valid_std)
