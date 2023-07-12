import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config


class Downsample(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=4,dropout=False,strides=2,padding=1,padding_mode='reflect', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.strides = strides
        self.padding = padding
        self.karnel_size = kernel_size
        self.dropout = dropout
        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels,self.output_channels,self.karnel_size,self.strides,self.padding,padding_mode=padding_mode),
            nn.BatchNorm2d(self.output_channels),
            nn.LeakyReLU(0.2)
        )
        if self.dropout:
            self.model.add_module('dropout',nn.Dropout(0.5))
    
    def forward(self,x):
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self,input_channels,features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initial_down = nn.Sequential(
            nn.Conv2d(input_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Downsample(features,features*2) 
        self.down2 = Downsample(features*2,features*4) 
        self.down3 = Downsample(features*4,features*8) 
        self.down4 = Downsample(features*8,features*8,padding_mode='zeros') 
        self.down5 = Downsample(features*8,features*8,padding_mode='zeros') 
        self.down6 = Downsample(features*8,1,padding_mode='zeros')

    def forward(self,x,y):
        x = torch.cat([x,y],1)  #(9x256x256)
        x = self.initial_down(x) # 128
        x = self.down1(x)   #64
        x = self.down2(x)   #32
        x = self.down3(x)   #16
        x = self.down4(x)   #8
        x = self.down6(x)   #4
        return x
    
def test():
    x = torch.randn((1,1,512,512))
    y = torch.randn((1,2,512,512))
    model = Discriminator(input_channels=3, features=64)
    preds = model(x,y)
    print(preds.shape)



if __name__ == "__main__":
    test()