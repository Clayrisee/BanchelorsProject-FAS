import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import os

class FASDataset(Dataset):
    def __init__(self, root_dir, csv_file, map_size=7, transform=None, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.map_size = map_size
        self.transform = transform
        
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99
    
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