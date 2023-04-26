import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/..')

from emu_Nx2pt.utils import get_config_from_yaml
from emu_Nx2pt.mlp_emulator import MLP_Emulator

def main(args=None):
    '''
        Usage:
            run a test (on yoda)
            >> cd /home/hhg/Research/emu_Nx2pt/repo/emulator_Nx2pt/
            >> python3 scripts/train_MLP_emulator.py --config configs/mlp_test.yaml > experiments/logs/mlp_test.log
            >> python3 scripts/train_MLP_emulator.py --config configs/mlp_run0.yaml > experiments/logs/mlp_run0.log
            >> python3 scripts/train_MLP_emulator.py --config configs/parallelMLP_run0.yaml > experiments/logs/parallelMLP_run0.log
            >> python3 scripts/train_MLP_emulator.py --config configs/parallelMLP_run2.yaml > experiments/logs/parallelMLP_run2.log
    '''
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='config file for training settings')

    opt = parser.parse_args()
    print(opt)

    config = get_config_from_yaml(opt.config)
    emu = MLP_Emulator(config=config)
    emu.train()
    #avg_test_chi2 = emu.test(input_dataloader)

if __name__ == '__main__':
    main()