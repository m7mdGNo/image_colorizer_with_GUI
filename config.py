import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "coco_dataset/"
VAL_DIR = "dataset/"
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc_resnet18.pt"
CHECKPOINT_GEN = "gen_resnet18.pt"
