import os, random, numpy as np, copy

from torch.utils.data import Dataset
import torch


def seq_collate(data):

    (past_traj, future_traj) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    class_ = past_traj[:,:,:,-1]
    future_traj = torch.stack(future_traj,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'seq': 'nba',
    }

    return data

class NBADataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, training=True
    ):
        super(NBADataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if training:
            data_root = 'datasets/nba/train_no_aug.npy'
        else:
            data_root = 'datasets/nba/test.npy'

        self.trajs = np.load(data_root)
        self.trajs /= (94/28) # Turn to meters
        # print(self.trajs)
        #if training:
        #    self.trajs = self.trajs[:32500]
        #else:
        #    self.trajs = self.trajs[:12500]

        self.batch_len = len(self.trajs)
        # print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)
        # print("before",self.traj_abs.shape)
        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        # print("after",self/.traj_abs.shape)

        self.traj_norm = self.traj_norm.permute(0,2,1,3)
        # print("size of traj shape",self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # traj_abs is a tensor of 205x11x15x2 
        # here 250 is the number of trainig data
        # 11 palayers are there
        # 15 is the total length of the observerd
        # 2 are the cordinates of the player
        # index is getting one data instance
        past_traj = self.traj_abs[index, :, :self.obs_len, :]
        future_traj = self.traj_abs[index, :, self.obs_len:, :]
        out = [past_traj, future_traj]
        return out
