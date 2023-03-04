import os
import numpy as np
import pandas as pd
import pickle
import torch
import time
import multiprocessing
import cosmolike_libs_RomanxCMB_10x2pt_datav as clike

dir_data = '/home/hhg/Research/emu_Nx2pt/data/'
filename = dir_data+'pco_train_1000000.pkl'
dir_out = '/home/hhg/Research/emu_Nx2pt/data/train_1M'

with open(filename, 'rb') as handle:
    pco_samples = pickle.load(handle)

## ----------------------------
## ------ Init Cosmolike ------
## ----------------------------

ROMAN, ROMAN_WIDE = 1, 0
CMB_SO, CMB_S4 = 1, 0
ONESAMPLE = 0

# -------------------------------------------------

if (ROMAN):
    if (CMB_SO):
        full_fid_datav = '10x2pt_RomanxSO'
    else:
        full_fid_datav = '10x2pt_RomanxS4'
if (ROMAN_WIDE):
    if (CMB_SO):
        full_fid_datav = '10x2pt_RomanWidexSO'
    else:
        full_fid_datav = '10x2pt_RomanWidexS4'

if (ONESAMPLE):
    full_fid_datav = full_fid_datav + '_1sample'
    
full_mask = full_fid_datav + '_mask.txt'

# -------------------------------------------------

which_cmb = "so_Y5" # default
# RomanxSO / S4
if (ROMAN):
    source_z = 'zdistri_WFIRST_LSST_lensing_fine_bin_norm'
    lens_z = 'zdistri_WFIRST_LSST_clustering_fine_bin_norm'
    sigma_z_shear = 0.01
    sigma_z_clustering = 0.01
    if (CMB_SO):
        survey_designation = "RomanxSO"
    else:
        survey_designation = "RomanxS4"
        which_cmb = "s4"
    tomo_binning_source = "source_std"
    tomo_binning_lens = "WF_SN10"
    lmax_shear = 4000.0

# RomanWidexSO / S4
if (ROMAN_WIDE):
    source_z = 'zdistri_WFIRST_LSST_lensing_fine_bin_norm'
    lens_z = 'zdistri_WFIRST_LSST_clustering_fine_bin_norm'
    sigma_z_shear = 0.02
    sigma_z_clustering = 0.02
    if (CMB_SO):
        survey_designation = "RomanWidexSO"
    else:
        survey_designation = "RomanWidexS4"
        which_cmb = "s4"
    tomo_binning_source = "source_std"
    tomo_binning_lens = "WF_SN10"
    lmax_shear = 4000.0

if (ONESAMPLE):
    lens_z = source_z
    sigma_z_clustering = sigma_z_shear
    survey_designation = survey_designation + '_1sample'
    tomo_binning_lens = "lens=src"

file_source_z = os.path.join(clike.dirname, "zdistris/",source_z)
file_lens_z = os.path.join(clike.dirname, "zdistris/",lens_z)

# -------------------------------------------------
## not used for datav, just placeholder - don't delete!!
shear_prior=0.003
delta_z_prior_shear=0.001
delta_z_prior_clustering=0.001
sigma_z_prior_shear=0.003
sigma_z_prior_clustering=0.003
nsource_table=51.0  
nlens_table=66.0
area_table=2000.0
# -------------------------------------------------

clike.initcosmo("halomodel".encode('utf-8'))
#clike.initbins(25,20.0,7979.0,lmax_shear,21.0,10,10)  # with scale cut
clike.initbins(25,20.0,7979.0,7970.0,0.0,10,10)       # without scale cut

clike.initpriors(shear_prior, sigma_z_shear, delta_z_prior_shear, sigma_z_prior_shear, sigma_z_clustering, delta_z_prior_clustering, sigma_z_prior_clustering)
clike.initsurvey(survey_designation.encode('utf-8'), nsource_table, nlens_table, area_table)
clike.initgalaxies(file_source_z.encode('utf-8'), file_lens_z.encode('utf-8'), "gaussian".encode('utf-8'), "gaussian".encode('utf-8'), tomo_binning_source.encode('utf-8'), tomo_binning_lens.encode('utf-8'))
clike.initia("NLA_z".encode('utf-8'), "none".encode('utf-8'))

clike.initprobes("10x2pt".encode('utf-8'))
clike.initcmb(which_cmb.encode('utf-8'))
clike.initfb(1)

## ---------------------------------------
## --------- Nuisance Parameters ---------
## ---------------------------------------

inp = clike.InputNuisanceParams()

if (ROMAN or ROMAN_WIDE): # Roman lens sample, gbias = 1.3 + 0.1*i, bin index i=0~9
    inp.bias[:] = [1.3 + 0.1*i for i in range(10)]
    if ROMAN:
        inp.source_z_s = 0.01
        inp.lens_z_s = 0.01
        src_z_s_fid, lens_z_s_fid = 0.01, 0.01
    else:
        inp.source_z_s = 0.02
        inp.lens_z_s = 0.02
        src_z_s_fid, lens_z_s_fid = 0.02, 0.02
        
    if (ONESAMPLE): # gbias use gold sample formula
        inp.lens_z_s = inp.source_z_s
        lens_z_s_fid = src_z_s_fid
        inp.bias[:] = [1.166664, 1.403981, 1.573795, 1.744325, 1.925937, 2.131001, 2.370211, 2.651007, 3.036247, 4.556622]


inp.source_z_bias[:] = [0 for _ in range(10)]
inp.lens_z_bias[:] = [0 for _ in range(10)]
inp.shear_m[:] = [0 for _ in range(10)]
#inp.A_ia = 0.5
#inp.eta_ia = 0.0
#inp.gas[:] = [1.17, 0.6, 14., 1., 0.03, 12.5, 1.2, 6.5, 0.752, 0., 0.]

## ---------------------------------------
## ------ Fixed Cosmology Parameters -----
## ---------------------------------------

MGSigma = MGmu = 0

## ---------------------------------------
## ------ Load Cov, Mask, Cal invL  ------
## ---------------------------------------

file_cov  = dir_data + 'cov3500.pkl'
file_mask = dir_data + '10x2pt_RomanxSO_fid_mask.txt'

with open(file_cov, 'rb') as handle:
    cov_full = pickle.load(handle)

mask_float = np.loadtxt(file_mask)[:, 1]
mask = mask_float.astype(bool)

cov_masked = cov_full[mask][:, mask]

# --- Compute invL ---

L = np.linalg.cholesky(cov_masked)
invL = np.linalg.inv(L)


## --------------------------------------------------
## ------ Start the for loop to generate dataVs -----
## --------------------------------------------------

def process_sample(i):
    Omega_m = pco_samples['Omega_m'][i]
    sigma_8 = pco_samples['sigma_8'][i]
    n_s     = pco_samples['n_s'][i]
    w0      = pco_samples['w0'][i]
    wa      = pco_samples['wa'][i]
    Omega_b = pco_samples['Omega_b'][i]
    h       = pco_samples['h0'][i]

    A_ia   =  pco_samples['A_ia'][i]
    eta_ia =  pco_samples['eta_ia'][i]
    
    gas_0  = pco_samples['gas_0'][i]   # Gamma
    gas_1  = pco_samples['gas_1'][i]   # beta
    gas_2  = pco_samples['gas_2'][i]   # log10(M0)
    gas_3  = pco_samples['gas_3'][i]   # alpha
    gas_7  = pco_samples['gas_7'][i]   # log10(Tw)
    gas_8  = pco_samples['gas_8'][i]   # f_H
    gas_9  = pco_samples['gas_9'][i]   # epsilon_1
    gas_10 = pco_samples['gas_10'][i]  # epsilon_2
    
    icp = clike.InputCosmologyParams(Omega_m, sigma_8, n_s, w0, wa, Omega_b, h, MGSigma, MGmu)

    inp.A_ia   = A_ia
    inp.eta_ia = eta_ia
    inp.gas[:] = [gas_0, gas_1, gas_2, gas_3, 0.03, 12.5, 1.2, gas_7, gas_8, gas_9, gas_10]
    
    datav_ptr = clike.get_data_vector(icp, inp)
    Ndata = 3500 # clike.get_Ndata(datav_ptr)
    dataV = np.array([datav_ptr[j] for j in range(Ndata)])

    # transform dataV -> dataT
    dataVp = invL@dataV[mask]
    dataT = torch.from_numpy(dataVp).float()
    torch.save(dataT, dir_out+f'/dataT_{i}.pt')

if __name__ == '__main__':
    
    start = time.time()
    
    Nsamples = len(pco_samples['Omega_m']) #1000000
    with multiprocessing.Pool() as pool:
        #pool.map(process_sample, range(0, 10))
        #pool.map(process_sample, range(0, 100000))
        pool.map(process_sample, range(100000, 300000))
        #pool.map(process_sample, range(300000, 400000))
        #pool.map(process_sample, range(400000, 500000))
        #pool.map(process_sample, range(500000, 600000))
    
    end = time.time()
    escape = (end-start)/3600.

    print(f'Total Time required to generate 0.2M dataVs: {escape} hr.')