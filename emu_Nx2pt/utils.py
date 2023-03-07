import yaml
from easydict import EasyDict
import torch
import torch.nn as nn

class ChiSquare(nn.Module):
    '''The loss function used when training emulator: chi2 of predicted & actual data vectors'''
    def __init__(self):
        super().__init__()

    def forward(self, predicted, actual):
        chi2 = ((predicted - actual) ** 2).sum() # mse = ((predicted - actual) ** 2).mean()
        return chi2

def get_config_from_yaml(file_yaml):
    '''Get the config from a yaml file
        Args:
            file_yaml: path to the config yaml file
        Return:
            config (EasyDict)
    '''

    with open(file_yaml, 'r') as file_config:
        try:
            config = EasyDict(yaml.safe_load(file_config))
            return config
        except ValueError:
            print("INVALID yaml file format.")
            exit(-1)


def display_layer_dimensions(net, input_dim):
    '''Display the output dimension of each layer in the given network
        Parameters :
            net (nn.Module) : network model
            input_dim (tuple) : e.g. (1, 3, 224, 224)
    '''
    device = next(net.parameters()).device
    X = torch.randn(input_dim).to(device)

    print('Input shape:\t\t', X.shape)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)


if __name__ == '__main__':
    config = get_config_from_yaml('../configs/mlp_test.yaml')
