import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import os
from data.check_type_image import is_image_file

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
        print(self.img_paths)
        print(len(self.img_paths))
        self.class_to_idx = {
            'spoof':0,
            'real':1
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
                if is_image_file(img_path):
                    img_path = os.path.join(root_dir, class_folder, img_path)
                    img_status = self.validate_image(img_path) # to_check image corrupted or not
                    if img_status == True:
                        img_paths.append(img_path) 
                    else:
                        continue
                else:
                    continue
        return img_paths, classes

    def class_to_idx(self, class_list):
        return {value:key for key,value in class_list.items()}
    
    def validate_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            return True
        except:
            return False

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        # print(img_path)
        label = img_path.split('/')[3]
        label = self.class_to_idx[label]
        if label == 1:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)
        else:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)

        if self.transform:
            img = self.transform(img)
        return img, mask, label




