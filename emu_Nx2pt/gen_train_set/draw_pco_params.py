import numpy as np
from pyDOE import lhs
import time
import argparse
import pickle

def draw_lhs_samples(Npts, par_lim, criterion='cm', par_keys=None):
    '''
    criterion: a string that tells lhs how to sample the points (default: None, which simply randomizes the points within the intervals):
        “center” or “c”: center the points within the sampling intervals
        “maximin” or “m”: maximize the minimum distance between points, but place the point in a randomized location within its interval
        “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
        “correlation” or “corr”: minimize the maximum correlation coefficient
    '''

    if not par_keys:
        par_keys = par_lim.keys()

    Ndim = len(par_lim)

    domain = np.zeros((2, Ndim))
    for i, key in enumerate(par_keys):
        domain[0][i] = par_lim[key][0]
        domain[1][i] = par_lim[key][1]

    #start = time.time()
    samples = lhs(Ndim, samples=Npts, criterion=criterion)
    #end = time.time()
    #period = (end - start)/60.
    #print(f'lhs finished: {period} mins')

    samples = domain[0] + samples * (domain[1] - domain[0])

    return samples


def main(args=None):
    '''
        Usage:
            run a test (on genie)
            >> cd /home/hhg/Research/emu_Nx2pt/repo/emulator_Nx2pt/
            >> python3 emu_Nx2pt/gen_train_set/draw_pco_params.py 100
            >> python3 emu_Nx2pt/gen_train_set/draw_pco_params.py 200000
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('Npts', help='Number of points to draw on lhs')

    args = parser.parse_args()
    Npts = int(args.Npts)
    print(f'--- Drawing {Npts} cosmology parameters ---')

    par_lim = {
        'Omega_m': [0.2, 0.4],
        'sigma_8': [0.8, 0.85],
        'Omega_b': [0.042, 0.050],
        'n_s': [0.9, 1.02],
        'h': [0.6, 0.8],
    }

    pco_samples = draw_lhs_samples(Npts=Npts, par_lim=par_lim, par_keys=['Omega_m', 'sigma_8', 'Omega_b', 'n_s', 'h'])

    data_dir = '/home/hhg/Research/emu_Nx2pt/data/'
    filename = data_dir+f'pco_m8bnh_{Npts}.pkl'
    print(f'--- Output pco_samples to: ---', filename)

    with open(filename, 'wb') as handle:
        pickle.dump(pco_samples, handle)


if __name__ == '__main__':
    main()
