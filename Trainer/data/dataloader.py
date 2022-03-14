
from data.base import DataModuleBase
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import FASDataset
from utils.custom_transforms import RandomGammaCorrection

class LivenessDataModule(DataModuleBase):
    def __init__(self, cfg):
        
        super().__init__(cfg)

        self.train_bs = cfg['train']['batch_size']
        self.val_bs = cfg['val']['batch_size']
        self.test_bs = cfg['test']['batch_size']
        self.map_size = cfg['model']['map_size']

        self.train_transforms = transforms.Compose([
            # RandomGammaCorrection(max_gamma=cfg['dataset']['augmentation']['gamma_correction'][1],
            #                 min_gamma=cfg['dataset']['augmentation']['gamma_correction'][0]),
            transforms.RandomHorizontalFlip() if cfg['dataset']['augmentation']['horizontal_flip'] else None,
            transforms.Resize(cfg['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['std'])
        ])

        self.test_val_transforms = transforms.Compose([
            transforms.Resize(cfg['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['std'])
        ])

        self.prepare_dataset(cfg=cfg)
    
    def prepare_dataset(self, cfg):

        self.train_set = FASDataset(
            root_dir=cfg['dataset']['root'],
            csv_file=cfg['dataset']['train_set'],
            map_size=cfg['model']['map_size'],
            transform=self.train_transforms,
            smoothing=cfg['train']['smoothing']
            )

        self.val_set = FASDataset(
            root_dir=cfg['dataset']['root'],
            csv_file=cfg['dataset']['val_set'],
            map_size=cfg['model']['map_size'],
            transform=self.test_val_transforms,
            smoothing=cfg['val']['smoothing']
            )
   
        self.test_set = FASDataset(
            root_dir=cfg['dataset']['root'],
            csv_file=cfg['dataset']['test_set'],
            map_size=cfg['model']['map_size'],
            transform=self.test_val_transforms,
            smoothing=cfg['test']['smoothing']
            )

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=2
        )
        self.train_dl = DataLoader(self.train_set, **kwargs)
        return self.train_dl

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=2
        )
        self.val_dl = DataLoader(self.val_set, **kwargs)
        return self.val_dl

    def test_dataloader(self):
        kwargs = dict(
            batch_size=self.test_bs,
            shuffle=True,
            num_workers=2
        )
        self.test_dl = DataLoader(self.test_set, **kwargs)
        return self.test_dl


# class LivenessDataset(LightningDataModule):
#     def __init__(self, train_csv, val_csv, test_csv, img_size, 
#                 batch_size=64):
#         super().__init__()
#         self.train_csv = train_csv
#         self.val_csv = val_csv
#         self.test_csv = test_csv
#         self.data = None
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.__load_and_transform()
        
#     def __load_and_transform(self):
#         train_val_transform = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(10),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         test_transform = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         if self.val_csv == None:
#             train_val_data = PixWiseDataset(self.train_csv, transform=train_val_transform)
#             dataset = train_val_data.dataset()
#             test_dataset = PixWiseDataset(self.test_csv, transform=test_transform)
#             val_size = int(len(dataset) * 0.2)
#             train_size = len(dataset) - val_size
#             self.train_set, self.val_set = random_split(dataset,(train_size, val_size))
#             self.test_set = test_dataset.dataset()

#         else:
#             # Adjust
#             self.train_set = datasets.ImageFolder(self.train_folder, transform=train_val_transform)
#             self.data = self.train_set
#             self.val_set = datasets.ImageFolder(self.val_folder, transform=train_val_transform)
#             self.test_set = datasets.ImageFolder(self.test_folder, transform=test_transform)
#             self.classes = self.train_set.classes
    
#     def train_dataloader(self):
#         kwargs = dict(
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=2
#         )
#         self.train_dl = DataLoader(self.train_set, **kwargs)
#         return self.train_dl
    
#     def val_dataloader(self):
#         kwargs = dict(
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=2
#         )
#         self.val_dl = DataLoader(self.val_set, **kwargs)
#         return self.val_dl
    
#     def test_dataloader(self, batch_size=376):
#         kwargs = dict(
#             batch_size=batch_size,
#             num_workers=2
#         )
#         self.test_dl = DataLoader(self.test_set, **kwargs)
#         return self.test_dl