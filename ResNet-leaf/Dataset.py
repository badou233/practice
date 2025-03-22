import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class LeafDataset(Dataset):
    def __init__(self, csv_data, root_dir, transform=None, label_to_index=None):
        self.csv_data = csv_data
        self.root_dir = root_dir
        self.transform = transform
        self.label_to_index = label_to_index

    def __getitem__(self, idx):
        img_name = self.csv_data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        if 'label' in self.csv_data.columns:
            label_str = self.csv_data.iloc[idx, 1]
            label = self.label_to_index[label_str]
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        else:
            return image, img_name

    def __len__(self):
        return len(self.csv_data)