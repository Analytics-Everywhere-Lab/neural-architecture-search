# Computes the NASWOT score of an architecture - NOT NEEDED

import random
import numpy as np
import torch
import os
from utils import get_score_func, add_dropout, init_network
# from cifar10loader import trainloader

from XAutoDL.xautodl.models import get_cell_based_tiny_net
from nats_bench import create

# os.environ['CUDA_VISIBLE_DEVICES'] = '' # CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #GPU

BATCHSIZE = 64
# sigma = 0.05 # default from NASWOT
# init = '' # default from NASWOT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()



def return_score(network, trainloader):
    # scores = np.zeros(len(arch_idx))
    scores = 0
    try:
        # add_dropout(network, sigma)
        # init_network(network, init)
        network.K = np.zeros((BATCHSIZE, BATCHSIZE))
        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):      # inputs can come as e.g. (img, label)
                    inp = inp[0]                # take just img
                inp = inp.view(inp.size(0), -1) # reshape into a 2D tensor of shape (batch_size, -1)
                x = (inp > 0).float()           # convert inp into binary tensor x with 1.0 where inp>0, 0.0 otherwise
                K = x @ x.t()                   # dot product for similarity of active
                K2 = (1.-x) @ (1.-x.t())        # dot product for similarity of inactive
                # print(K.size(), K2.size(), network.K.shape) # Make sure batch size of trainloader matches that used here
                network.K += K.cpu().numpy() + K2.cpu().numpy()
            except:
                pass

            
        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

            
        for name, module in network.named_modules():
            if 'ReLU' in str(type(module)):
                #hooks[name] = module.register_forward_hook(counting_hook)
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)

        network = network.to(device)
        s = []
        maxofn = 1 # defines the number of times it is calculated
        for j in range(maxofn):
            data_iterator = iter(trainloader)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, device)
            network(x2.to(device))
            s.append(get_score_func('hook_logdet')(network.K, target))
        scores = np.mean(s)
    except Exception as e:
        print(e)
        scores = np.nan
    return(scores)
