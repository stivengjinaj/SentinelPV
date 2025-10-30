import os

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import torch
import math


from .senseiver_dataset import NOAA, pipe, plume, porous, cylinder

from .senser_loc import ( cylinder_16_sensors, 
                         cylinder_8_sensors, 
                         cylinder_4BC_sensors,
                         sea_n_sensors,
                         sensors_3D
                         )



import datetime

from einops import rearrange, repeat

from torch.utils.data import DataLoader,Dataset


def PositionalEncoder(image_shape,num_frequency_bands,max_frequencies=None):
    
    *spatial_shape, _ = image_shape
   
    coords = [ torch.linspace(-1, 1, steps=s) for s in spatial_shape ]
    pos = torch.stack(torch.meshgrid(*coords), dim=len(spatial_shape)) 
    
    encodings = []
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    frequencies = [ torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                                              for max_freq in max_frequencies ]
    
    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(pos[..., i:i+1] * frequencies_i[None, ...])

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    enc = torch.cat(encodings, dim=-1)
    enc = rearrange(enc, "... c -> (...) c")

    return enc






def load_data(dataset_name, num_sensors, seed=123):
    
    if dataset_name == 'cylinder':
        data = cylinder()
        
        if num_sensors == 16:
            x_sens, y_sens = cylinder_16_sensors()
            
        if num_sensors == 8:
            x_sens, y_sens = cylinder_8_sensors()
            
        if num_sensors == 4:
            x_sens, y_sens = cylinder_4_sensors()
            
        if num_sensors == 4444:
            x_sens, y_sens = cylinder_4BC_sensors()
            
            
        else:
            x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
    
            
    elif dataset_name == 'sea':
        data, time = NOAA()
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
        return torch.as_tensor( data, dtype=torch.float ), time, x_sens, y_sens
        
    elif dataset_name == 'pipe':
       data = pipe()
       x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
       
    elif dataset_name == 'plume':
        data = plume()
        data = data[None,:,:,:,None]
        x_sens, *y_sens = sensors_3D(data, num_sensors, seed)
        
    elif dataset_name == 'pore':
        data = porous()
        data = data[:,:,:,:,None]
        x_sens, *y_sens = sensors_3D(data, num_sensors, seed)
       
    else:
        #raise NameError('Unknown dataset')
        print(f'The dataset_name {dataset_name} was not provided\n')
        print('************WARNING************')
        print('*******************************\n')
        print('Creating a dummy dataset\n')
        print('************WARNING************')
        print('*******************************\n')
        data = np.random.rand(1000,150,75,1)
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
        
    print(f'Data size {data.shape}\n')
    
    return torch.as_tensor( data, dtype=torch.float ), x_sens, y_sens
    


def senseiver_dataset(data_config):
    
    data_name   = data_config['data_name']
    num_sensors = data_config['num_sensors']
    seed        = data_config['seed']
    sample_train_dataset = data_config['sample_train_dataset']
    
    data, x_sens, y_sens = load_data(data_name, num_sensors, seed)
    
    total_frames, *image_size, im_ch = data.shape
    
    training_frames = data_config['training_frames']
    
    if sample_train_dataset:
        if seed:
            torch.manual_seed(seed)
        train_ind = torch.randperm(data.shape[0])[:training_frames]
        
        data = data[train_ind]
    
    else:
        data = data[training_frames:]
    
    # sensor coordinates
    sensors = torch.zeros(data.shape[1:-1])
    
    if len(sensors.shape) == 2:
        sensors[x_sens,y_sens] = 1
    elif len(sensors.shape) == 3: # 3D images
        sensors[x_sens,y_sens[0],y_sens[1]] = 1
        
    sensors = sensors.unsqueeze(0).unsqueeze(-1).repeat(data.shape[0],1,1,1)
    
    print(sensors.shape, data.shape)
    
    dataset = torch.utils.data.TensorDataset(sensors, data)
    return dataset   
       
     
    
