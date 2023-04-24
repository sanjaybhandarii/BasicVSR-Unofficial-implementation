
import torch
from mmedit.models.losses import CharbonnierLoss

LOAD_MODEL = True
SAVE_MODEL = True

CHECKPOINT_BEST_LOAD = "best/best_model.pth"

CHECKPOINT_LATEST_SAVE = "latest/latest_model.pth"

CHECKPOINT_BEST_SAVE = "best/best_model.pth"



TRAIN_HR_PATH = 'reds/train'

VAL_HR_PATH = 'reds/test'

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 8
NUM_WORKERS = 4
HIGH_RES = 256
LOW_RES = HIGH_RES//4
IMG_CHANNELS = 3

charbonnier_loss = CharbonnierLoss(eps=1e-8)
