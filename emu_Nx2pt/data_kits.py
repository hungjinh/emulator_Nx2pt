import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset

class dataTDataset(Dataset):
    
    def __init__(self, df_pco, dir_dataT, startID, endID, transform=None):
        self.dir_dataT = dir_dataT
        self.df_pco = df_pco
        self.startID = startID
        self.endID = endID
        
    def __getitem__(self, index):
        dataT = torch.load( os.path.join(self.dir_dataT, 'dataT_'+str(index)+'.pt') )[self.startID:self.endID]
        pco = torch.from_numpy(self.df_pco.iloc[index].values).float()
        return index, pco, dataT

    def __len__(self):
        return len(self.df_pco)