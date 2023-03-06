import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset

class dataTDataset(Dataset):
    
    def __init__(self, IDs, df_pco, dir_dataT, transform=None):
        self.IDs = IDs
        self.dir_dataT = dir_dataT
        self.df_pco = df_pco
        
    def __getitem__(self, index):
        pcoID = self.IDs[index]
        dataT = torch.load( os.path.join(self.dir_dataT, 'dataT_'+str(pcoID)+'.pt') )
        pco = torch.from_numpy(self.df_pco.loc[[pcoID]].values[0]).float()
        return pco, dataT
    
    def __len__(self):
        return len(self.IDs)