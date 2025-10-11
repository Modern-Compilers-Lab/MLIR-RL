from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from utils.data_collector import OfflineDataset


dt = OfflineDataset(save_dir="offline_dataset",fname="offline_dataset_online_ppo.npz")


a = dt.load()

print(a[''][100:150])