import yaml
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from psutil._common import bytes2human

from transformers import T5Model


# Used for NASWOT Implementation. Can delete if NASWOT score won't be used
def hooklogdet(K, labels=None):
    # print(K)
    s, ld = np.linalg.slogdet(K)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()

_scores = {
    'hook_logdet': hooklogdet,
    'random': random_score
}

def get_score_func(score_name):
    return _scores[score_name]

def read_mem(nt):
    value = ''
    # print(nt)
    for name in nt._fields:
        if name == 'used':
            value = getattr(nt, name)
            # print(bytes2human(value))
            # value = value
            # value = value / (1024 ** 2)
            
        # print('%-10s : %7s' % (name.capitalize(), value))
    # return value / (1024 * 1024)
    # print(int(value) / (1024 ** 2))
    return int(value) / (1024 ** 2)

def calculate_network_size(network):
    total_params = 0
    for param in network.parameters():
        param_memory = param.numel() * param.element_size()
        total_params += param_memory


### Evaluate the necessity of these

class DropChannel(torch.nn.Module):
    def __init__(self, p, mod):
        super(DropChannel, self).__init__()
        self.mod = mod
        self.p = p
    def forward(self, s0, s1, droppath):
        ret = self.mod(s0, s1, droppath)
        return ret


class DropConnect(torch.nn.Module):
    def __init__(self, p):
        super(DropConnect, self).__init__()
        self.p = p
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        dim1 = inputs.shape[2]
        dim2 = inputs.shape[3]
        channel_size = inputs.shape[1]
        keep_prob = 1 - self.p
        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, channel_size, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output    

def add_dropout(network, p, prefix=''):
    #p = 0.5
    for attr_str in dir(network):
        target_attr = getattr(network, attr_str)
        if isinstance(target_attr, torch.nn.Conv2d):
            setattr(network, attr_str, torch.nn.Sequential(target_attr, DropConnect(p)))
        
    for n, ch in list(network.named_children()):
        #print(f'{prefix}add_dropout {n}')
        if isinstance(ch, torch.nn.Conv2d):
            setattr(network, n, torch.nn.Sequential(ch, DropConnect(p)))
        else:
            add_dropout(ch, p, prefix + '\t')


def orth_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.orthogonal_(m.weight)

def uni_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight)

def uni2_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight, -1., 1.)

def uni3_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight, -.5, .5)

def norm_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.norm_(m.weight)

def eye_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.eye_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.dirac_(m.weight)

def init_network(network, init):
    if init == 'orthogonal':
        network.apply(orth_init)
    elif init == 'uniform':
        print('uniform')
        network.apply(uni_init)
    elif init == 'uniform2':
        network.apply(uni2_init)
    elif init == 'uniform3':
        network.apply(uni3_init)
    elif init == 'normal':
        network.apply(norm_init)
    elif init == 'identity':
        network.apply(eye_init)