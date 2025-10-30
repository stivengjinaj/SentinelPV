import numpy as np
import h5py
import pickle
import torch


def NOAA():
   
    f = h5py.File('/data/sensor/sst_weekly.mat','r') 
    sst = np.nan_to_num( np.array(f['sst']) )
    
    num_frames = 1914

    sea = np.zeros((num_frames,180,360,1))
    for t in range(num_frames):
        sea[t,:,:,0] = sst[t,:].reshape(180,360,order='F')
    sea /= sea.max()
    
    time = torch.arange(1,1915)
    time = (time % 52)
    return sea, time



def pipe():
    
   with open("/data/sensor/ch_2Dxysec.pickle", 'rb') as f:
       pipe = pickle.load(f)
       pipe /= np.abs(pipe).max()
   return pipe


def cylinder():
    
    with open('/data/sensor/Cy_Taira.pickle', 'rb') as f:
        cyl = pickle.load(f)
    return cyl
    

def plume():
    with h5py.File('Data/Plume/concentration.h5', "r") as f:
        plume_3D = f['cs']
        plume_3D = np.array(plume_3D)
        plume_3D /= plume_3D.max()
    return plume_3D


def porous():
    with h5py.File('Data/Pore/rho_1.h5', "r") as f:
        pore = f['rho'][:]
    return pore
    
def isotropic3D():
    with h5py.File('Isotropic/scalarHIT_fields100.h5', "r") as f:
        return np.array(f['fields'])
