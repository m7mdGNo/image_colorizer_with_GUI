import torch
import torch.nn as nn
import torch.nn.functional as F
import config as conf

class ColorizationLoss(nn.Module):
    def __init__(self):
        super(ColorizationLoss, self).__init__()

    def forward(self, outputs, targets):
        # Split the outputs into L (grayscale) and ab (color) channels
        outputs_L, outputs_ab = torch.split(outputs, [1, 2], dim=1)
        targets_L, targets_ab = torch.split(targets, [1, 2], dim=1)

        # Compute the classification loss (L channel)
        classification_loss = F.binary_cross_entropy_with_logits(outputs_L, targets_L)

        # Compute the regression loss (ab channels)
        regression_loss = F.smooth_l1_loss(outputs_ab, targets_ab)

        # Combine the losses
        loss = classification_loss + 0.5 * regression_loss

        return loss


# Define the number of bins for quantizing the ab color space
Q = 313

# Define the weight for the regression loss
lambd = 0.5

# Define the temperature for the softmax function
T = 0.38

# Define a function to convert an ab color to a bin index
def ab2bin(a, b):
    # Normalize the ab values to [-1, 1]
    a = (a + 128) / 255 * 2 - 1
    b = (b + 128) / 255 * 2 - 1
    # Quantize the ab values using a grid of size sqrt(Q) x sqrt(Q)
    q_a = torch.floor((a + 1) / 2 * torch.sqrt(Q))
    q_b = torch.floor((b + 1) / 2 * torch.sqrt(Q))
    # Compute the bin index by concatenating the quantized values
    bin_idx = q_a * torch.sqrt(Q) + q_b
    return bin_idx.long()

# Define a function to convert a bin index to an ab color
def bin2ab(bin_idx):
    # Compute the quantized ab values by dividing and modding the bin index
    q_a = bin_idx // torch.sqrt(Q)
    q_b = bin_idx % torch.sqrt(Q)
    # De-normalize the ab values to [0, 255]
    a = (q_a / torch.sqrt(Q) * 2 - 1) * 255 / 2 + 128
    b = (q_b / torch.sqrt(Q) * 2 - 1) * 255 / 2 + 128
    return a, b

# Define a function to compute the softmax cross-entropy loss
def softmax_cross_entropy_loss(logits, labels):
    # Apply softmax with temperature to the logits
    probs = F.softmax(logits / T, dim=1)
    # Convert the labels to one-hot vectors
    one_hot = F.one_hot(labels, num_classes=Q)
    # Compute the cross-entropy loss
    loss = -torch.mean(torch.sum(one_hot * torch.log(probs + 1e-8), dim=1))
    return loss

# Define a function to compute the Euclidean distance loss
def euclidean_distance_loss(logits, labels):
    # Compute the mean color prediction by taking the expectation over the logits
    mean_color_pred = torch.sum(logits * bin2ab(torch.arange(Q)), dim=1)
    # Compute the mean color ground truth by averaging the labels
    mean_color_gt = torch.mean(bin2ab(labels), dim=1)
    # Compute the Euclidean distance loss
    loss = torch.mean(torch.norm(mean_color_pred - mean_color_gt, dim=1))
    return loss

# Define a function to compute the final loss
def colorful_loss(logits, labels):
    # Compute the classification loss
    class_loss = softmax_cross_entropy_loss(logits, labels)
    # Compute the regression loss
    reg_loss = euclidean_distance_loss(logits, labels)
    # Compute the final loss as a weighted sum of the two losses
    final_loss = class_loss + lambd * reg_loss
    return final_loss


    
if __name__ == "__main__":
    x = torch.randn((1,2,256,256)).to(conf.DEVICE)
    y = torch.randn((1,2,256,256)).to(conf.DEVICE)
    # loss = ColorfulLoss()
    loss_value = colorful_loss(x,y)
    print(loss_value)