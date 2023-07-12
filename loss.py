import torch
import torch.nn as nn
import pytorch_ssim  # import the MS-SSIM library
import torchvision.models as models
import torch.nn.functional as F
import config as conf
from Discriminator import Discriminator


class ColorizationLoss(nn.Module):
    def __init__(self, lambda_g=0.1, lambda_s=0.1):
        super(ColorizationLoss, self).__init__()
        self.lambda_g = lambda_g
        self.lambda_s = lambda_s
    
    def forward(self,ab_pred, ab_true):
        # Color error loss
        le = F.mse_loss(ab_pred, ab_true)
        
        # Class distribution loss
        ls = F.kl_div(F.log_softmax(ab_pred, dim=1), ab_true)
        
        # Total loss
        loss = self.lambda_g * le + self.lambda_s * ls
        
        return loss


class HistLoss(nn.Module):
    def __init__(self):
        super(HistLoss, self).__init__()
    
    def forward(self, pred, target):
        # pred and target are batched histograms with shape (batch_size, k)
        # convert to probabilities by normalizing along the k dimension
        pred_prob = F.softmax(pred, dim=1)
        target_prob = target.float() / target.sum(dim=1, keepdim=True).float()
        
        # compute KL divergence
        loss = F.kl_div(pred_prob.log(), target_prob, reduction='batchmean')
        
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).to(conf.DEVICE).features[:35]
        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        with torch.no_grad():
            input_vgg = self.vgg(input)
            target_vgg = self.vgg(target)
            loss = self.criterion(input_vgg, target_vgg)
        return loss
    

if __name__ == "__main__":
    x = torch.randn((16,3,256,256)).to(conf.DEVICE)
    y = torch.randn((16,3,256,256)).to(conf.DEVICE)
    loss = PerceptualLoss()
    loss_value = loss(x,y)
    print(loss_value)
    # y_hat = torch.randn((1,2,256,256),dtype=torch.float32)
    # D = Discriminator(3,64)
    # loss = generator_loss(x,y,y_hat,D)
    # print(loss)
    # loss = discriminator_hinge_loss(D,x,y,y_hat)
    # print(loss)