from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import cv2
import torchvision
from torchvision import transforms
import glob
import numpy as np
import torch
import config as conf
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

    
class ColorzationDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = list(glob.glob(img_paths +  f'/*.jpg')+ glob.glob(img_paths + '/*.jpeg') + glob.glob(img_paths + '/*.png'))
        self.transform = transforms.Compose([
                transforms.Resize((conf.IMAGE_SIZE,conf.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2),                
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        img = self.transform(img)
        img = np.array(img)
        img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        l = img_lab[:,:,0]
        l = np.expand_dims(l,2)
        ab = img_lab[:,:,1:]
        l = l.astype('float32')/255.0
        ab = ab.astype('float32')/255.0

        L = transforms.ToTensor()(l)
        ab = transforms.ToTensor()(ab)
        return L,ab


def test():
    input_path = 'coco_dataset/'
    dataset = ColorzationDataset(input_path)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        x = x.squeeze().numpy()
        x = np.expand_dims(x,axis=2)
        y = y.squeeze().permute(1, 2, 0).numpy()

        lab_img = np.concatenate([x,y],axis=2)
        bgr = cv2.cvtColor(lab_img,cv2.COLOR_LAB2BGR)

        cv2.imshow('gray',np.uint8((x*255)))
        cv2.imshow('color',np.uint8((bgr*255)))
        if cv2.waitKey(0) == 27:
            break

if __name__ == "__main__":
    test()