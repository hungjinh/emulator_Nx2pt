import os
import pickle
import torch

class BaseTrainer():
    def __init__(self, config):

        self._setup(config)
    
    def _setup(self, config):

        print('\n------ Parameters ------\n')
        for key in config:
            setattr(self, key, config[key])
            print(f'{key} :', config[key])
        
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & self.cuda
        self.device = torch.device(f'cuda:{self.gpu_device}' if self.cuda else "cpu")

        print('\n------ Device ------\n')
        print(self.device)

        self.dir_exp = os.path.join(self.dir_output, self.exp_name)
        self.dir_checkpoints = os.path.join(self.dir_exp, 'checkpoints')
        self.file_trainInfo = os.path.join(self.dir_exp, 'trainInfo.pkl')
    
    def _init_storage(self):
        '''initialize storage dictionary and directory to save training information'''

        # ------ create storage directory ------
        print('\n------ Create experiment directory ------\n')
        try:
            os.makedirs(self.dir_exp)
        except (FileExistsError, OSError) as err:
            raise FileExistsError(
                f'Default save directory {self.dir_exp} already exit. Change exp_name!') from err
        print(f'Training information will be stored at :\n \t {self.dir_exp}')

        # ------ create 'checkpoints' directory to store 'self.statInfo_{epochID}.pth'
        os.makedirs(self.dir_checkpoints)

        # ------ trainInfo ------
        save_key = ['train_loss', 'valid_loss', 'epoch_train_loss', 'epoch_valid_loss', 'lr']
        self.trainInfo = {key:[] for key in save_key}

        # ------ stateInfo ------
        self.statInfo = {}
    
    def _save_checkpoint(self, epochID):
        
        self.trainInfo['best_epochID'] = self.best_epochID
        self.trainInfo['best_model_wts'] = self.best_model_wts
        torch.save(self.trainInfo, self.file_trainInfo)


        self.statInfo = {
            'epoch': epochID,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

        outfile_statInfo = os.path.join(self.dir_checkpoints, f'stateInfo_{epochID}.pth')
        torch.save(self.statInfo, outfile_statInfo)

    def load_checkpoint(self, epochID=None):
        
        trainInfo = torch.load(self.file_trainInfo)

        if epochID:
            statInfo = torch.load(os.path.join(self.dir_checkpoints, f'stateInfo_{epochID}.pth'))
            
            return trainInfo, statInfo
        else:
            return trainInfo