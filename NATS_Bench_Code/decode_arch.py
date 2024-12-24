# Attempts to decode the string representation of an architecture. Work on this later

import torch
from torch import nn
from torchvision.models import vgg16 
import yaml
import math 
from utils import parse_network

BATCH_SIZE = 64

class DecodedModel(nn.Module):
    def __init__(self, config):
        super(DecodedModel, self).__init__()

        self.layers = nn.ModuleList() # ModuleList allows appending layers. Preferred when appending with loop
        in_channels = config['input_channels']
        feat_size = config['input_W']

        for layer in config['layers']:
            layer_type = layer['type']

            if layer_type == 'CONVOLUTION':
                out_channels = layer['conv_param']['out_channels']
                kernel_size = layer['conv_param']['kernel_size']
                padding = layer['conv_param']['padding']

                conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
                self.layers.append(conv_layer)
                in_channels = out_channels # Save for next layer
                feat_size = self.get_map_size(feat_size, kernel_size, padding, 1)
            
            elif layer_type == 'RELU':
                self.layers.append(nn.ReLU(inplace=True))
            
            elif layer_type == 'POOLING':
                pool_type = layer['pooling_param']['pool']
                kernel_size = layer['pooling_param']['kernel_size']
                stride = layer['pooling_param']['stride']

                if pool_type == 'MAXPOOL':
                    pool_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
                else:
                    pool_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
                
                self.layers.append(pool_layer)
                feat_size = self.get_map_size(feat_size, kern_size=kernel_size, pad=padding, st=stride)

            elif layer_type == 'FLATTEN':
                self.flatten = nn.Flatten()
                self.layers.append(self.flatten)
                feat_size -= 1
                in_channels = feat_size * feat_size * in_channels # 7 * 7 * 512. Calculated here so it is only done once


            elif layer_type == 'INNER_PRODUCT':
                # print(in_channels)
                out_channels = layer['inner_product_param']['out_channels']
                fc_layer = nn.Linear(in_channels, out_channels)
                self.layers.append(fc_layer)
                in_channels = out_channels
            
            elif layer_type == 'DROPOUT':
                dropout_ratio = layer['dropout_param']['dropout_ratio']
                dropout_layer = nn.Dropout(p=dropout_ratio)
                self.layers.append(dropout_layer)

            elif layer_type == 'SOFTMAX':
                self.layers.append(nn.Softmax(dim=0))

        self.layers = nn.Sequential(*self.layers)

    def get_map_size(self, feat_size, kern_size, pad, st):
        return math.floor(((feat_size-kern_size+ (2*pad)) / st) + 1)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # print(f'Layer {idx} Output shape: {x.shape}')
        return x

  
# config = parse_network('vgg.yaml')    

# model = DecodedModel(config)
# print(model)

# input_tensor = torch.ones(BATCH_SIZE, 3, 224, 224)
# output = model(input_tensor)
# vgg_model = vgg16()
# # print(vgg_model)
# vgg_output = vgg_model(input_tensor)
# print(f'Output shape: {output.shape}')
# print(f'VGG Output shape: {vgg_output.shape}')

# # Compare the outputs
# are_outputs_equal = torch.equal(output, vgg_output)
# print(f"Are the outputs exactly equal? {are_outputs_equal}")
# are_outputs_close = torch.allclose(output, vgg_output, rtol=1e-05, atol=1e-08)
# print(f"Are the outputs close (considering floating-point precision)? {are_outputs_close}")