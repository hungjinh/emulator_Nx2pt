import os
import time
import copy
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from emu_Nx2pt.base import BaseTrainer
from emu_Nx2pt.utils import ChiSquare, display_layer_dimensions
from emu_Nx2pt.models.mlp import MLP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim

from emu_Nx2pt.data_kits import dataTDataset


class MLP_Emulator(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)
        self._prepare_data()
        self._get_Npco_Nout()
        self._build_model()
        self._define_loss()
        self._init_optimizer()

        self.compute_L()
    
    def _prepare_data(self):

        with open(self.filename_pco, 'rb') as handle:
            pco_samples = pickle.load(handle)

        self.df_pco = pd.DataFrame(pco_samples).iloc[self.startID:self.endID]
    
        self.Nsamples = len(self.df_pco['Omega_m'])
        sampleID = list(range(self.Nsamples))

        trainID, validID = train_test_split(sampleID, train_size=self.f_train, random_state=self.seed)

        self.dataset = {} 
        self.dataset['train'] = dataTDataset(trainID, self.df_pco, self.dir_dataT)
        self.dataset['valid'] = dataTDataset(validID, self.df_pco, self.dir_dataT)

        self.dataloader = {}
        self.dataloader['train'] = DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True)
        self.dataloader['valid'] = DataLoader(self.dataset['valid'], batch_size=self.batch_size, shuffle=True)

        print('\n------ Prepare Data ------\n')
        for key in ['train', 'valid']:
            print(f'Number of {key} samples: {len(self.dataset[key])} ({len(self.dataloader[key])} batches)')
    
    def _get_Npco_Nout(self):
        '''Getting the dimensions of inputs and outputs from the training examples'''
        validSamples = iter(self.dataloader['valid'])
        pco, datav = validSamples.next()
        self.Nout = len(datav[0])
        self.Npco = pco.shape[1]
    
    def _build_model(self):
        self.model = MLP(self.Npco, self.Nout, self.Nblocks, self.Nhidden).to(self.device)

        print('\n------ Build Model ------\n')
        print(self.model, '\n')
        print('Number of trainable parameters:', sum(param.numel() for param in self.model.parameters() if param.requires_grad))

    def _define_loss(self):
        self.criterion = ChiSquare()
    
    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def _train_one_epoch(self):
        
        for phase in ['train', 'valid']:
            if phase == 'train': 
                self.model.train()
            else:
                self.model.eval()
            
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.dataloader[phase]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                
                self.trainInfo[f'{phase}_loss'].append( loss.item()/inputs.size(0) )

                running_loss += loss.item()

            # ------ End training the epoch in a train or valid phase ------
            epoch_loss = running_loss / len(self.dataset[phase])
            self.trainInfo[f'epoch_{phase}_loss'].append(epoch_loss)
            
            print(f'\t{phase} avg_chi2: {epoch_loss:.2f}', end='')

            if phase == 'train':
                self.scheduler.step()
            
            if phase == 'valid':
                if epoch_loss < self.min_valid_loss: # -> deep copy the model
                    self.best_epochID = self.curr_epochID
                    self.min_valid_loss = epoch_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
        # ------ End training the ephch in both train & valid phases ------
        self.trainInfo['lr'].append(self.scheduler.get_last_lr()[0]) # save lr / epoch

    def train(self):

        self._init_storage()
        self.curr_epochID = 0
        self.best_epochID = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.min_valid_loss = np.inf

        for epochID in range(self.num_epochs):
            self.curr_epochID = epochID

            print(f'--- Epoch {epochID+1}/{self.num_epochs} ---')
            
            since = time.time()
            self._train_one_epoch()
            self._save_checkpoint(epochID)

            time_cost = time.time() - since
            print(f'\tTime: {time_cost//60:.0f}m {time_cost%60:.0f}s')    

            if self.curr_epochID - self.best_epochID >= self.early_stop_threshold:
                print(f'Early stopping... (Model did not imporve after {self.early_stop_threshold} epochs)')
                break
        
        print(f'\nMinimum (epoch-averaged) validation loss reached at epoch {self.best_epochID+1}.')
    
    def gen_dataT(self, pco):
        '''Generate decorrelated dataT given pco'''

        with torch.no_grad():
            emu_dataT = self.model(pco).to('cpu').numpy()
        
        return emu_dataT
    
    def compute_L(self):
        with open(self.file_cov, 'rb') as handle:
            cov_full = pickle.load(handle)

        mask_float = np.loadtxt(self.file_mask)[:, 1]
        mask = mask_float.astype(bool)

        cov_masked = cov_full[mask][:, mask]

        self.L = np.linalg.cholesky(cov_masked)
        self.invL = np.linalg.inv(self.L)

    def gen_dataV(self, pco):
        
        dataT = self.gen_dataT(pco)
        dataV = self.L@dataT

        return dataV

