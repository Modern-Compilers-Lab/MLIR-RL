import torch
from utils.log import print_info

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == __package__:
    print_info("Using device:", device.type)
