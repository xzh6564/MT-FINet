import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class GastricDataset(Dataset):
    def __init__(self, csv_file, img_dir1, img_dir2, transform1=None, transform2=None):
        self.df = pd.read_csv(csv_file, encoding='gb18030')
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['num'])

        path1 = os.path.join(self.img_dir1, img_name + '.jpg')
        path2 = os.path.join(self.img_dir2, img_name + '.png')

        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        if self.transform1:
            img1 = self.transform1(img1)
        if self.transform2:
            img2 = self.transform2(img2)

        image = torch.cat([img1, img2], dim=0)

        clinical = torch.tensor(
            row[['gender', 'age', 'tumor_size', 'ETE']].values.astype(float),
            dtype=torch.float32)

        cclnm = torch.tensor(row['Label1'], dtype=torch.float32)
        lclnm = torch.tensor(row['Label2'], dtype=torch.float32)

        return image, clinical, cclnm, lclnm, img_name
