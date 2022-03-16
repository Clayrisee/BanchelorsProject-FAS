import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DatasetFolder
import os

class FASDataset(Dataset):
    def __init__(self, root_dir, csv_file, map_size=7, transform=None, smoothing=True):
        super(FASDataset, self).__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.map_size = map_size
        self.transform = transform
        
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :

        img_name = self.data.iloc[index, 0]
        img_name = os.path.join(self.root_dir, img_name)

        img = Image.open(img_name)
        label = self.data.iloc[index, 1].astype(np.float32)
        label = np.expand_dims(label, axis=0)
        if label == 1:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)
        else:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)
        
        if self.transform:
            img = self.transform(img)
        
        return img, mask, label

class FASFolderDataset(Dataset):

    def __init__(self, root_dir, map_size=7, transform=None, smoothing=True):
        super(FASFolderDataset, self).__init__()
        self.img_paths, self.class_name = self._make_dataset(root_dir=root_dir)
        self.class_to_idx = {
            'spoof':0,
            'live':1
        }
        self.map_size=map_size
        self.transform = transform
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99

    def _make_dataset(self, root_dir):
        img_paths=[]
        classes= os.listdir(root_dir)
        for class_folder in classes:
            for img_path in os.listdir(os.path.join(root_dir, class_folder)):
                img_paths.append(os.path.join(root_dir, class_folder, img_path))
        return img_paths, classes

    def class_to_idx(self, class_list):
        return {value:key for key,value in class_list.items()}
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        label = img_path.split('/')[1]
        label = self.class_to_idx[label]
        if label == 1:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)
        else:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)

        if self.transform:
            img = self.transform(img)
        return img, mask, label



