import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 4
BATCH_SIZE = 64
PATH_TO_DATA = "./data"  # consider ./module as root
PATH_TO_MODEL = "./models/{}"  # usage: PATH_TO_MODEL.format(model_filename)
