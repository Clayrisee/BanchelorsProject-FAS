from models.layers.cdcn_layers import Conv2d_cd
import torch.nn as nn
import torch

class CDCN(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCN, self).__init__()

        self.conv1 =  nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )

        self.block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block2 =  nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )

        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

    def forward(self, x):
        # x_input = x
        x = self.conv1(x)

        x_block1 = self.block1(x) # x [128, 128, 128]
        x_block1_32x32 = self.downsample32x32(x_block1) # x [128, 32, 32]

        x_block2 = self.block2(x_block1) #  # x [128, 64, 64]	  
        x_block2_32x32 = self.downsample32x32(x_block2) # [128, 32, 32]

        x_block3 = self.block3(x_block2) #  x [128, 32, 32]
        x_block3_32x32 = self.downsample32x32(x_block3) # x [128, 32, 32]

        x_concat = torch.cat((x_block1_32x32, x_block2_32x32, x_block3_32x32)) # x [128 *3, 32, 32] (concat low, mid, and high result of feature extraction)

        x = self.lastconv1(x_concat) # x[128, 32, 32] turn 128 * 3 to 128 again
        x = self.lastconv2(x) #  x[64, 32, 32]
        x = self.lastconv3(x) # x [1, 32, 32] result map

        map_x = x.squeeze(1) 
        score = torch.mean(map_x, axis=(1,2))
        return map_x, score
