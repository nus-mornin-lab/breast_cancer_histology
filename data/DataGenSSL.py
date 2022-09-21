import os, sys, glob, csv, random, re, operator

from toolz import *
from toolz.curried import *

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

from PIL import Image
#########################################################################################################
class DataGen(Dataset):
        
    def __init__(self, dataDir, transform = identity, train = True) :

        self.dataDir   = dataDir
        self.train     = train
        self.transform = transform
        self.files     = parse(dataDir, train)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        
        file = self.files[i]

        patch = np.load(file)
        
        # patch : [3,D,256,256]

        C,D,W,H = patch.shape
        patch = torch.tensor(patch)
        
        # some slices have more than 17 slices, I normalize them to 17
        if "DCONV" in self.dataDir :
            if D != 16:
                patch = patch.unsqueeze(0)            
                patch = torch.nn.functional.interpolate(patch, size = (16, W, H))
                patch = patch.squeeze(0)
        else :
            if D != 17:
                patch = patch.unsqueeze(0)            
                patch = torch.nn.functional.interpolate(patch, size = (17, W, H))
                patch = patch.squeeze(0)
        
        return self.transform(patch)
#########################################################################################################    
def parse(dataDir, train):    
    return glob.glob(f"{dataDir}/*.npy")
    
#########################################################################################################    

if __name__ == '__main__':
    
    from easydict import EasyDict
    from matplotlib import pyplot as plt
    import time
        
    def normalise(x):
        
        val_max = 1
        val_min = 0
        
        _min = torch.tensor(x.min(dim = 1)[0].min(dim = 1)[0]).unsqueeze(-1).unsqueeze(-1)
        _max = torch.tensor(x.max(dim = 1)[0].max(dim = 1)[0]).unsqueeze(-1).unsqueeze(-1)
        
        y = ( x - _min ) / ( _max - _min )
        
        return y        
    
    D = 17
    config = EasyDict()
    config.dataDir = "./datasets/processed/flatten/*"
    config.train = True
    
    gen = DataGen(config.dataDir, train = config.train)
    
    for i in range(len(gen)):
                
        # volume : [3 17 256 256]
        volume = gen.__getitem__(i)            
        fig, axs = plt.subplots(nrows = 1, ncols = D, figsize=(30,30*D))
        
        for i, ax in enumerate(axs):            
            slice = normalise(volume[:, i, ...])            
            ax.imshow(slice.permute(1,2,0))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
        
        time.time(2)
                        
            
            
            
            