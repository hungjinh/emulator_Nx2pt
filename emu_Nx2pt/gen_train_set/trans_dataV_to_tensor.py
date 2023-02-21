import torch
import time
import numpy as np
import pickle
import multiprocessing
import argparse


def transform_dataV(i, dir_dataV, dir_out, mask, invL):
    dataV = np.loadtxt(dir_dataV+f'/10x2pt_emu_{i}')[:, 1]
    dataVp = invL@dataV[mask]
    dataT = torch.from_numpy(dataVp).float()
    torch.save(dataT, dir_out+f'/dataT_{i}.pt')

def main(args=None):
    '''
         Usage:
         >> cd /home/hhg/Research/emu_Nx2pt/repo/emulator_Nx2pt/
         >> python3 emu_Nx2pt/gen_train_set/trans_dataV_to_tensor.py train_300_raw train_300 0 2
         >> python3 emu_Nx2pt/gen_train_set/trans_dataV_to_tensor.py train_1M_raw train_1M 0 100000
         >> python3 emu_Nx2pt/gen_train_set/trans_dataV_to_tensor.py train_1M_raw train_1M 100000 200000
    '''

    dir_data = '/home/hhg/Research/emu_Nx2pt/data/'

    parser = argparse.ArgumentParser()
    parser.add_argument('dir_dataV', help='the director name of raw dataV')
    parser.add_argument('dir_out', help='the director name of output dataV tensors')
    parser.add_argument('startID', help='startID of the dataV to be transformed')
    parser.add_argument('endID', help='endID of the dataV to be transformed')

    args = parser.parse_args()
    dir_dataV = dir_data + args.dir_dataV
    dir_out = dir_data + args.dir_out
    startID = int(args.startID)
    endID = int(args.endID)

    # --- 1. Load Cov, mask ---
    file_cov = dir_data + 'cov3500.pkl'
    file_mask = dir_data + 'train_1M_raw/10x2pt_emu_0_mask.txt'

    with open(file_cov, 'rb') as handle:
        cov_full = pickle.load(handle)

    mask_float = np.loadtxt(file_mask)[:, 1]
    mask = mask_float.astype(bool)

    cov_masked = cov_full[mask][:, mask]

    # --- 2. Compute invL ---
    L = np.linalg.cholesky(cov_masked)
    invL = np.linalg.inv(L)
    
    # --- 3. Transform np.arr dataV to torch.tensor ---

    Ncpu = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=Ncpu) as pool:
        params = [(i, dir_dataV, dir_out, mask, invL) for i in range(startID, endID)]
        pool.starmap(transform_dataV, params)



if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    escape = (end-start)/60.
    print(f'Total Time Cost : {escape} mins')
