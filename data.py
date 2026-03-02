"""
This file contains all the custom Dataset classes of the approaches used
and a custom LightningDataModule class.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import lightning as L


def tabular_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Standartize a two dimension tensor by column (feature).

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Standartized tensor.
    """
    
    sigma, mu = torch.std_mean(tensor, dim=0)
    
    output = (tensor - mu)/sigma
    return output

class FF_Dataset(Dataset):
    def __init__(self, xez: Tuple[pd.core.frame.DataFrame],
                 extra: pd.core.frame.DataFrame = None,
                 p=1, norm=False):
        
        z = torch.tensor(xez[2].values)
        self.z = z.reshape((z.numel(), 1))
        
        if xez[1] is None:
            x = torch.tensor(xez[0].fillna(0).values)           
            e = torch.zeros(x.size()[0], x.size()[1])
        else:
            e = torch.tensor(xez[1].fillna(0).values)
            
        x = torch.tensor(xez[0].fillna(0).values)
        
        x = x.repeat(p, 1)
        e = e.repeat(p, 1)
        
        self.z = self.z.repeat(p, 1)
            
        if p == 1:
            if xez[1] is not None:
                self.covariables = torch.cat((x, e), dim=1)
            else:
                self.covariables = x
        else:
            self.covariables = torch.normal(x, e)
            
        if extra is not None:
            extra = torch.tensor(extra.values).repeat(p, 1)
            self.covariables = torch.cat((self.covariables, extra), dim=1)
            
    def __getitem__(self, idx):
        return self.covariables[idx], self.z[idx]

    def __len__(self):
        return len(self.z)
    
def make_vectors(x: torch.Tensor, e: torch.Tensor,
                 version=['no', 'with', 'stack']) -> torch.Tensor:

    n = len(x)
    p = len(x[0])
    
    x = torch.reshape(x, (n, 1, p))
    e = torch.reshape(e, (n, 1, p))

    if version == 'no':
        return x

    if version == 'with':
        return torch.cat((x, e), dim=2)
    if version == 'stack':
        return torch.cat((x, e), dim=1)
    
def matrix_1d_norm(tensor: torch.Tensor) -> torch.Tensor:
    sigma, mu = torch.std_mean(tensor, dim=(0, 2))
    
    n_channels = len(tensor[0])
    
    for i in range(n_channels):
        tensor[:, i, :] = (tensor[:, i, :] - mu[i])/sigma[i]
    return tensor
    
class CNN1D_Dataset(Dataset):
    def __init__(self, xez: Tuple[pd.core.frame.DataFrame],
                 extra: pd.core.frame.DataFrame = None,
                 version=['no', 'with', 'stack'],
                 p=1, norm=False):

        if p != 1 and version != 'no':
            raise ValueError("super and not 'no'")

        z = torch.tensor(xez[2].values)
        self.z = z.reshape((z.numel(), 1))

        x = torch.tensor(xez[0].fillna(0).values)
        e = torch.tensor(xez[1].fillna(0).values)
        x = x.repeat(p, 1)
        e = e.repeat(p, 1)
        
        self.z = self.z.repeat(p, 1)
        
        if p == 1:
            self._1d = make_vectors(x, e, version=version)
        else:
            new_x = torch.normal(x, e)
            self._1d = make_vectors(new_x, e, version='no')
        
        if norm is not None:
            self._1d = matrix_1d_norm(self._1d)
            
        if extra is not None:
            extra = torch.tensor(extra.values)
            extra = torch.reshape(extra, (len(extra), 1, extra.size()[1]))
            extra = extra.repeat(p, self._1d.size()[1], 1)
            self._1d = torch.cat((self._1d, extra), dim=2)

    def __getitem__(self, idx):
        return self._1d[idx], self.z[idx]

    def __len__(self):
        return len(self.z)
    

def make_matrices(
    x: torch.Tensor, e: torch.Tensor,
    K: int, t_inf: float, t_sup: float) -> torch.Tensor:
    n = len(x)
    p = len(x[0])
    
    xbar = x.nanmean(dim=1)
    xbar = xbar.reshape(n, 1, 1, 1)
    xbar = xbar.repeat(1, 1, K, p)
    
    x = x.reshape(n, 1, 1, p)
    x = torch.nan_to_num(x, float('inf'))
    x = x.repeat(1, 1, K, 1)
    
    e = e.reshape(n, 1, 1, p)
    e = torch.nan_to_num(e, 1e-6)
    e = e.repeat(1, 1, K, 1)
    
    t = -t_inf + (t_inf + t_sup)/(2*K) + (t_inf + t_sup)/(K)*torch.tensor(range(K))
    t = t.reshape(1, 1, K, 1)
    t = t.repeat(n, 1, 1, p)
    
    mu_p = t + xbar
    
    normal = torch.distributions.normal.Normal(loc=mu_p, scale=e)
    
    return normal.log_prob(x).exp()

def matrix_2d_norm(tensor: torch.Tensor) -> torch.Tensor:
    sigma, mu = torch.std_mean(tensor, dim=(0, 2, 3))
    
    n_channels = len(tensor[0])
    
    for i in range(n_channels):
        tensor[:, i, :, :] = (tensor[:, i, :, :] - mu[i])/sigma[i]
    return tensor
    
class CNN2D_Dataset(Dataset): 
    def __init__(self, xez: Tuple[pd.core.frame.DataFrame],
                 extra: pd.core.frame.DataFrame = None,
                 K: int=None, t_inf: float=None, t_sup: float=None,
                 norm=False):
        
        z = torch.tensor(xez[2].values)
        self.z = z.reshape((z.numel(), 1))

        x = torch.tensor(xez[0].fillna(0).values)
        e = torch.tensor(xez[1].fillna(0).values)
        
        self._2d = make_matrices(x, e, K, t_inf, t_sup)

        if norm is not None:
            self._2d = matrix_2d_norm(self._2d)
            
        if extra is not None:
            extra = torch.tensor(extra.values)
            extra = torch.reshape(extra, (len(extra), 1, 1, extra.size()[1]))
            extra = extra.repeat(1, x.size()[1], x.size()[3], 1)
            self._2d = torch.cat((self._2d, extra), dim=3)

    def __getitem__(self, idx):
        return self._2d[idx], self.z[idx]

    def __len__(self):
        return len(self.z)
    
class Custom_DataModule(L.LightningDataModule):
    def __init__(self, *args, xez: Tuple[pd.core.frame.DataFrame],
                 extra: pd.core.frame.DataFrame = None,
                 dataset_class,
                 split_size_or_sections=[0.5, 0.25, 0.25], seed: int=2023,
                 batch_size:int = None, num_workers: int=None,
                 **kwargs):
        super().__init__(*args)
        
        if xez[1] is None:
            data = dataset_class((xez[0], xez[1], xez[2]), extra, **kwargs)
        else:
            data = dataset_class((xez[0], xez[1], xez[2]), extra, **kwargs)
            
        
        data = dataset_class((xez[0], xez[1], xez[2]), extra, **kwargs)
        
        n = len(data)
        
        shuffled_ind = torch.randperm(n, generator=torch.manual_seed(seed))
        
        if len(split_size_or_sections) == 2:
            ind_train, ind_val = torch.split(shuffled_ind, split_size_or_sections)
            ind_test = ind_val
            
        if len(split_size_or_sections) == 3:
            ind_train, ind_val, ind_test = torch.split(shuffled_ind, split_size_or_sections)
        
        self.train = Subset(data, ind_train)
        self.val = Subset(data, ind_val)
        self.test = Subset(data, ind_test)
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

        
