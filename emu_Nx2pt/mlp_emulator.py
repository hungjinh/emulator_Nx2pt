import os
import time
import copy
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from emu_Nx2pt.base import BaseTrainer
from emu_Nx2pt.utils import ChiSquare, display_layer_dimensions
from emu_Nx2pt.models.mlp import MLP, MLP_Res, ParallelMicroNets, AttentionBasedMLP

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
        self._build_model(file_model_state=self.file_model_state)
        self._define_loss()
        self._init_optimizer()

        self.compute_L()
    
    def _prepare_data(self):
        
        pco = {}
        self.df_pco = {}
        self.dataset = {} 
        self.dataloader = {}

        for key in ['train', 'valid']:

            with open(self.file_pco[key], 'rb') as handle:
                pco[key] = pickle.load(handle)
            
            self.df_pco[key] = pd.DataFrame(pco[key])

            self.dataset[key] = dataTDataset(self.df_pco[key], self.dir_dataT[key], self.startID, self.endID)

            self.dataloader[key] = DataLoader(self.dataset[key], batch_size=self.batch_size, shuffle=True, num_workers=self.workers)


        print('\n------ Prepare Data ------\n')
        for key in ['train', 'valid']:
            print(f'Number of {key} samples: {len(self.dataset[key])} ({len(self.dataloader[key])} batches)')
    
    def _get_Npco_Nout(self):
        '''Getting the dimensions of inputs and outputs from the training examples'''
        validSamples = iter(self.dataloader['valid'])
        _, pco, datav = next(validSamples)
        #self.output_size = len(datav[0])
        self.input_size = pco.shape[1]
    
    def _build_model(self, file_model_state=None):
        
        if self.model_type == 'MLP':
            self.model = MLP(self.input_size, self.output_size, self.hidden_size, self.Nblocks, self.is_batchNorm).to(self.device)
        elif self.model_type == 'MLP_Res':
            self.model = MLP_Res(self.input_size, self.output_size, self.hidden_size, self.Nblocks, self.is_batchNorm, self.scale_factor).to(self.device)
        elif self.model_type == 'ParallelMicroNets':
            self.model = ParallelMicroNets(self.input_size, self.encode_size, self.hidden_size, self.output_size, self.Nblocks, self.scale_factor).to(self.device)
        elif self.model_type == 'AttentionBasedMLP':
            self.model = AttentionBasedMLP(self.input_size, self.output_size, self.hidden_size, self.Nblocks, self.Nseq, self.num_heads, self.mlp_ratio, self.scale_factor).to(self.device)


        print('\n------ Build Model ------\n')
        print(self.model, '\n')
        print('Number of trainable parameters:', sum(param.numel() for param in self.model.parameters() if param.requires_grad))

        if file_model_state:
            trainInfo = torch.load(file_model_state)
            print('... Load previously best trained parameters to model.')
            self.model.load_state_dict(trainInfo['best_model_wts'])
        else:
            print('... Initialized model with random parameters.')

    def _define_loss(self):
        self.criterion = ChiSquare()
    
    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def _train_one_epoch(self):
        
        self.model.train()
                    
        running_loss = 0.0
        for i, (_, inputs, labels) in enumerate(self.dataloader['train']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                
            self.trainInfo[f'train_loss'].append( loss.item()/inputs.size(0) )

            running_loss += loss.item()

        # ------ Finished the whole training epoch ------
        epoch_loss = running_loss / len(self.dataset['train'])
        self.trainInfo[f'epoch_train_loss'].append(epoch_loss)
            
        print(f'\ttrain avg_chi2: {epoch_loss:.2f}', end='')

        self.trainInfo['lr'].append(self.scheduler.get_last_lr()[0]) # save lr / epoch
        
        self.scheduler.step()


    def _valid_one_epoch(self):

        self.model.eval()

        running_loss = 0.0
        for i, (_, inputs, labels) in enumerate(self.dataloader['valid']):
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            self.trainInfo['valid_loss'].append( loss.item()/inputs.size(0) )

            running_loss += loss.item()

        # ------ Finished the whole validating epoch ------
        epoch_loss = running_loss / len(self.dataset['valid'])
        self.trainInfo['epoch_valid_loss'].append(epoch_loss)
        
        print(f'\tvalid avg_chi2: {epoch_loss:.2f}', end='')

        if epoch_loss < self.min_valid_loss: # -> deep copy the model
            self.best_epochID = self.curr_epochID
            self.min_valid_loss = epoch_loss
            self.best_model_wts = copy.deepcopy(self.model.state_dict())



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
            self._valid_one_epoch()

            self._save_checkpoint(epochID)

            time_cost = time.time() - since
            print(f'\tTime: {time_cost//60:.0f}m {time_cost%60:.0f}s')    

            if self.curr_epochID - self.best_epochID >= self.early_stop_threshold:
                print(f'Early stopping... (Model did not improve after {self.early_stop_threshold} epochs)')
                break
        
        print(f'\nMinimum (epoch-averaged) validation loss reached at epoch {self.best_epochID+1}.')
    
    def run_test_loop(self, dataloader):

        print('\n------ Run test loop ------\n')
        print(f'  Number of galaxies: {len(dataloader.dataset)} ({len(dataloader)} batches)')

        self.model.eval()
        running_loss = 0.0
        for _, inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
        
        avg_test_chi2 = running_loss / len(dataloader.dataset)

        print(f'  Average chi2 of the test set: {avg_test_chi2:.2f}')

        return avg_test_chi2
    
    def test(self, dataloader, epochID=None):
        '''Test the model at given epochID on the given dataset. 
            If epochID is not specified, use the best-trained model.
        '''

        if epochID:
            print('Test model with the model at epochID:', epochID)
            trainInfo, statInfo = self.load_checkpoint(epochID)
            self.model.load_state_dict(statInfo['model_state_dict'])
        else:
            trainInfo = self.load_checkpoint()
            print('Test model with the best-trained model at epochID:', trainInfo['best_epochID'])
            self.model.load_state_dict(trainInfo['best_model_wts'])
        
        avg_test_chi2 = self.run_test_loop(dataloader)

        return avg_test_chi2

    
    def gen_dataT(self, pco):
        '''Generate decorrelated dataT given pco

        Args:
            pco (torch.Tensor): cosmological parameters 
        
        Returns:
            np.array: emulated decorrelated dataT 
        '''

        if len(pco.shape) == 1:
            pco = pco.view((1, -1))

        with torch.no_grad():
            emu_dataT = self.model(pco).to('cpu').numpy()
        
        if emu_dataT.shape[0]==1:
            return emu_dataT[0]
        else:
            return emu_dataT
    
    def compute_L(self):
        with open(self.file_cov, 'rb') as handle:
            cov_full = pickle.load(handle)

        mask_float = np.loadtxt(self.file_mask)[:, 1]
        mask = mask_float.astype(bool)

        self.cov_masked = cov_full[mask][:, mask]
        self.cov_cut = self.cov_masked[self.startID:self.endID][:, self.startID:self.endID]

        self.L = np.linalg.cholesky(self.cov_cut)

        self.invL = np.linalg.inv(self.L)

    def gen_dataV(self, pco):
        '''Generate emulated dataV given pco

        Args:
            pco (torch.Tensor): cosmological parameters 
        
        Returns:
            np.array: emulated data vector (can directly compare with CosmoLike)
        '''
        
        dataT = self.gen_dataT(pco)
        dataV = self.L@dataT

        return dataV


