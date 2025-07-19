import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np
from pathlib import Path
import h5py
from einops import rearrange
import os
import torch.nn as nn
import random

### 1. 1D datasets
def get_data(data_dir, dataset_name, return_params=False):    
    if dataset_name == "ks":  # not used in the paper
        data_dir = f"{data_dir}/KS_small_L64"
        train_path = f"{data_dir}/KS_train_2400.h5"
        valid_path = f"{data_dir}/KS_valid.h5"
        test_path = f"{data_dir}/KS_test.h5"
        resolution = "pde_140-256"
        parameter = "nu"
    
    elif dataset_name == "wave_varying_boundary":  # used in the paper
        data_dir = f"{data_dir}/wave_varying_boundary"   
        train_path = f"{data_dir}/WE_train_WE3.h5"
        valid_path = f"{data_dir}/WE_valid_WE3.h5"
        test_path = f"{data_dir}/WE_test_WE3.h5"
        resolution = "pde_250-256"
        parameter = "c"
    
    elif dataset_name == "advection":  # used in the paper
        data_dir = f"{data_dir}/Advection"   
        train_path = f"{data_dir}/Advection_train_12000.h5"
        valid_path = f"{data_dir}/Advection_valid.h5"
        test_path = f"{data_dir}/Advection_test.h5"
        resolution = "pde_140-256"
        parameter = "nu"
    
    elif dataset_name == "burgers":  # not used in the paper
        data_dir = f"{data_dir}/mppde_small_forcing_large"   
        train_path = f"{data_dir}/CE_train_E2.h5"
        valid_path = f"{data_dir}/CE_valid_E2.h5"
        test_path = f"{data_dir}/CE_test_E2.h5"
        resolution = "pde_250-256"
        parameter = "beta"
    
    elif dataset_name == "burgers_nu_forcing":  # not used in the paper
        data_dir = f"{data_dir}/burgers_nu_forcing"   #WARNING
        train_path = f"{data_dir}/CE_train_E2.h5"
        valid_path = f"{data_dir}/CE_valid_E2.h5"
        test_path = f"{data_dir}/CE_test_E2.h5"
        resolution = "pde_250-256"
        parameter = "beta"
      
    elif dataset_name == "burgers_nu_forcing2": # used in the paper
        data_dir = f"{data_dir}/burgers_nu_forcing2"   #WARNING
        train_path = f"{data_dir}/CE_train_E2.h5"
        valid_path = f"{data_dir}/CE_valid_E2.h5"
        test_path = f"{data_dir}/CE_test_E2.h5"
        resolution = "pde_250-256"
        parameter = "beta"
        
    elif dataset_name == "heat":  # not used in the paper
        data_dir = f"{data_dir}/heat_nu_forcing" #heat_large   #WARNING
        train_path = f"{data_dir}/CE_train_E4.h5"
        valid_path = f"{data_dir}/CE_valid_E4.h5"
        test_path = f"{data_dir}/CE_test_E4.h5"
        resolution = "pde_250-256"
        parameter = "beta"

    elif dataset_name == "heat2":  # used in the paper
        data_dir = f"{data_dir}/heat_nu_forcing2"   #WARNING
        train_path = f"{data_dir}/CE_train_E4.h5"
        valid_path = f"{data_dir}/CE_valid_E4.h5"
        test_path = f"{data_dir}/CE_test_E4.h5"
        resolution = "pde_250-256"
        parameter = "beta"
    
    elif dataset_name == "combined_equation":  # used in the paper
        data_dir = f"{data_dir}/combined_equation"
        u_tr = []
        u_te = []
        u_val = []
        for version in ["v1", "v2", "v3", "v4"]:   
            train_path = f"{data_dir}/{version}/CE_train_3000.h5"
            valid_path = f"{data_dir}/{version}/CE_valid.h5"
            test_path = f"{data_dir}/{version}/CE_test.h5"
            f_train = h5py.File(train_path, 'r')
            f_val = h5py.File(valid_path, 'r')
            f_te = h5py.File(test_path, 'r')
            _u_tr = f_train['train']["pde_140-256"][()][..., None]
            _u_val = f_val['valid']["pde_140-256"][()][..., None]
            _u_te = f_te['test']["pde_140-256"][()][..., None]
            u_tr.append(_u_tr)
            u_val.append(_u_val)
            u_te.append(_u_te)

        u_tr = np.concatenate(u_tr)
        u_val = np.concatenate(u_val)
        u_te = np.concatenate(u_te)

        u_tr = rearrange(u_tr, 'b t h c -> b c h t')
        u_val = rearrange(u_val, 'b t h c -> b c h t')
        u_te = rearrange(u_te, 'b t h c -> b c h t')

    
    if dataset_name not in ['combined_equation']:
        f_tr = h5py.File(train_path, 'r')
        f_val = h5py.File(valid_path, 'r')
        f_te = h5py.File(test_path, 'r')
        
        u_tr = f_tr['train'][resolution][()][..., None] # create channel dimension
        u_val = f_val['valid'][resolution][()][..., None]
        u_te = f_te['test'][resolution][()][..., None]

        param_tr = f_tr['train'][parameter][()][..., None]
        param_val = f_val['valid'][parameter][()][..., None]
        param_te = f_te['test'][parameter][()][..., None]
        
        u_tr = rearrange(u_tr, 'b t h c -> b c h t')
        u_val = rearrange(u_val, 'b t h c -> b c h t')
        u_te = rearrange(u_te, 'b t h c -> b c h t')

        if return_params:
            return u_tr, u_val, u_te, param_tr, param_val, param_te
        
    return u_tr, u_val, u_te

def get_data_ood(dataset_name, return_params=False, option=None):
    
    if dataset_name == "wave_varying_boundary":  # used in the paper
        data_dir = f"{data_dir}/wave_varying_boundary/ood/"   
        train_path = f"{data_dir}/WE_train_WE3.h5"
        valid_path = f"{data_dir}/WE_valid_WE3.h5"
        test_path = f"{data_dir}/WE_test_WE3.h5"
        resolution = "pde_250-256"
        parameter = "c"
    
    elif dataset_name == "advection":  # used in the paper
        data_dir = f"{data_dir}/Advection/ood/"   
        train_path = f"{data_dir}/Advection_train_10.h5"
        valid_path = f"{data_dir}/Advection_valid.h5"
        test_path = f"{data_dir}/Advection_test.h5"
        resolution = "pde_140-256"
        parameter = "nu"
      
    elif dataset_name == "burgers_nu_forcing2": # used in the paper
        if option is not None:
            data_dir = f"{data_dir}/burgers_nu_forcing2/ood_{option}/"   #WARNING
        else:
            data_dir = f"{data_dir}/burgers_nu_forcing2/ood"

        train_path = f"{data_dir}/CE_train_E2.h5"
        valid_path = f"{data_dir}/CE_valid_E2.h5"
        test_path = f"{data_dir}/CE_test_E2.h5"
        resolution = "pde_250-256"
        parameter = "beta"
        
    elif dataset_name == "heat2":  # used in the paper

        if option is not None:
            data_dir = f"{data_dir}/heat_nu_forcing2/ood_{option}/"   #WARNING
        else:
            data_dir = f"{data_dir}/heat_nu_forcing2/ood"

        train_path = f"{data_dir}/CE_train_E4.h5"
        valid_path = f"{data_dir}/CE_valid_E4.h5"
        test_path = f"{data_dir}/CE_test_E4.h5"
        resolution = "pde_250-256"
        parameter = "beta"
    
    elif dataset_name == "combined_equation":  # used in the paper
        data_dir = f"{data_dir}/combined_equation/ood/"
        u_tr = []
        u_te = []
        u_val = []   
        train_path = f"{data_dir}CE_train_10.h5"
        valid_path = f"{data_dir}CE_valid.h5"
        test_path = f"{data_dir}CE_test.h5"
        f_train = h5py.File(train_path, 'r')
        f_val = h5py.File(valid_path, 'r')
        f_te = h5py.File(test_path, 'r')
        _u_tr = f_train['train']["pde_140-256"][()][..., None]
        _u_val = f_val['valid']["pde_140-256"][()][..., None]
        _u_te = f_te['test']["pde_140-256"][()][..., None]
        u_tr.append(_u_tr)
        u_val.append(_u_val)
        u_te.append(_u_te)

        u_tr = np.concatenate(u_tr)
        u_val = np.concatenate(u_val)
        u_te = np.concatenate(u_te)

        u_tr = rearrange(u_tr, 'b t h c -> b c h t')
        u_val = rearrange(u_val, 'b t h c -> b c h t')
        u_te = rearrange(u_te, 'b t h c -> b c h t')

        return u_tr, u_val, u_te
    
    if dataset_name not in ['advection_diffusion', 'wave2d', 'ns2d']:
        f_tr = h5py.File(train_path, 'r')
        f_val = h5py.File(valid_path, 'r')
        f_te = h5py.File(test_path, 'r')
        
        u_tr = f_tr['train'][resolution][()][..., None] # create channel dimension
        u_val = f_val['valid'][resolution][()][..., None]
        u_te = f_te['test'][resolution][()][..., None]

        param_tr = f_tr['train'][parameter][()][..., None]
        param_val = f_val['valid'][parameter][()][..., None]
        param_te = f_te['test'][parameter][()][..., None]
        
        u_tr = rearrange(u_tr, 'b t h c -> b c h t')
        u_val = rearrange(u_val, 'b t h c -> b c h t')
        u_te = rearrange(u_te, 'b t h c -> b c h t')

        if return_params:
            return u_tr, u_val, u_te, param_tr, param_val, param_te
        
        return u_tr, u_val, u_te

class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, u, sub_t=1, slice_size=20):
        self.u = u 
        self.sub_t = sub_t
        self.slice_size = slice_size

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        images = self.u[idx, ..., ::self.sub_t]
        max_start_index = images.shape[-1] - self.slice_size
        if max_start_index < 0:
            raise ValueError("Slice size is larger than the sequence length.")
        start_index = np.random.randint(0, max_start_index + 1)
        images = images[..., start_index:start_index + self.slice_size]

        return torch.from_numpy(images).float()
    

class TemporalDatasetWithContext(torch.utils.data.Dataset):
    def __init__(self, u, sub_t=10, slice_size=10, num_context_trajectories=1):
        self.u = u 
        self.slice_size = slice_size
        self.num_context_trajectories = num_context_trajectories
        self.subsampling_t = sub_t
        self.random_t = False

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        sub_t = self.subsampling_t
        images = self.u[idx, ..., ::sub_t] # self.subsampling_t

        max_start_index = images.shape[-1] - self.slice_size
        if max_start_index < 0:
            raise ValueError("Slice size is larger than the sequence length.")
        start_index = np.random.randint(0, max_start_index + 1)
        images = images[..., start_index:start_index + self.slice_size]

        min_index = (idx // 10)*10 # we have 10 trajectories per environment
        max_index = min_index + 9

        context_images = []
        max_start_index = images.shape[-1] - self.slice_size # WARNING test ongoing
        ctx_idx_list = [i for i in range(min_index, max_index+1)] 
        ctx_idx_list.remove(idx)
        random.shuffle(ctx_idx_list)
        for j in range(self.num_context_trajectories):
            if self.random_t:
                start_index = np.random.randint(0, max_start_index + 1)  # WARNING test ongoing
            ctx_idx = ctx_idx_list[j]
            context_images.append(self.u[ctx_idx, ..., ::sub_t][..., start_index:start_index + self.slice_size])

        context_images = torch.stack(context_images)

        return images, context_images



class ReacDiffDataset(Dataset):
    def __init__(self, file_name, group, sub_t=1, slice_size=1):
        self.file_name = file_name
        self.group = group
        self.sub_t = sub_t
        self.slice_size=slice_size

        with h5py.File(self.file_name, 'r') as f:
            self.length = len(f[f'{self.group}/states'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_name, 'r') as f:
            state = torch.tensor(f[f'{self.group}/states'][idx], dtype=torch.float32)
            f_param = torch.tensor(f[f'{self.group}/f'][idx], dtype=torch.float32)
            k_param = torch.tensor(f[f'{self.group}/k'][idx], dtype=torch.float32)
            state = state[..., ::self.sub_t]
            max_start_index = state.shape[-1] - self.slice_size
            if max_start_index < 0:
                raise ValueError("Slice size is larger than the sequence length.")
            start_index = np.random.randint(0, max_start_index + 1)
            state = state[..., start_index:start_index + self.slice_size]
            
            seed = torch.tensor(f[f'{self.group}/seed'][idx], dtype=torch.float32)
        return state


class VortDataset(Dataset):
    def __init__(self, file_name, group, smooth=True, sub_t=1, slice_size=1):
        self.file_name = file_name
        self.group = group
        self.sub_t = sub_t
        self.slice_size = slice_size
        self.smooth = smooth
        self.scale = 20
        self.layer = nn.Conv2d(1, 1, 2, stride=2, padding_mode="circular", padding=0, bias=False)
        self.layer.weight = nn.Parameter(torch.ones(1, 1, 2, 2) / 4.0, requires_grad=False)

        print('smoothing', smooth, 'scale', self.scale)

        # Open HDF5 file once
        self.file = h5py.File(self.file_name, 'r')
        self.length = len(self.file[f'{self.group}/states'])

    def __del__(self):
        # Ensure the file is closed properly
        self.file.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        state = torch.tensor(self.file[f'{self.group}/states'][idx], dtype=torch.float32) / self.scale
        state = state[..., ::self.sub_t]
        if self.smooth:
            t = state.shape[-1]
            state = rearrange(state, 'c h w t -> t c h w')
            with torch.no_grad():
                state = self.layer(state)
            state = rearrange(state, 't c h w -> c h w t', t=t)

        max_start_index = state.shape[-1] - self.slice_size
        if max_start_index < 0:
            raise ValueError("Slice size is larger than the sequence length.")
        start_index = np.random.randint(0, max_start_index + 1)
        state = state[..., start_index:start_index + self.slice_size]

        nu = torch.tensor(self.file[f'{self.group}/mu'][idx], dtype=torch.float32)
        ic = self.file[f'{self.group}/ic'][idx].decode('utf-8')
        seed = torch.tensor(self.file[f'{self.group}/seed'][idx], dtype=torch.float32)
        return state


class WaveDataset(Dataset):
    def __init__(self, file_name, group, sub_t=1, slice_size=1):
        self.file_name = file_name
        self.group = group
        self.slice_size=slice_size
        self.sub_t = sub_t

        with h5py.File(self.file_name, 'r') as f:
            self.length = len(f[f'{self.group}/states'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_name, 'r') as f:
            state = torch.tensor(f[f'{self.group}/states'][idx], dtype=torch.float32)[0][None]
            state = state[..., ::self.sub_t]
            max_start_index = state.shape[-1] - self.slice_size
            if max_start_index < 0:
                raise ValueError("Slice size is larger than the sequence length.")
            start_index = np.random.randint(0, max_start_index + 1)
            state = state[..., start_index:start_index + self.slice_size]

            k = torch.tensor(f[f'{self.group}/k'][idx], dtype=torch.float32)
            c = torch.tensor(f[f'{self.group}/c'][idx], dtype=torch.float32)

            bc = f[f'{self.group}/bc'][idx].decode('utf-8')
            ic = f[f'{self.group}/ic'][idx].decode('utf-8')
            seed = torch.tensor(f[f'{self.group}/seed'][idx], dtype=torch.float32)
        return state
    
def load_wave2d(file_name, train_batch_size = 32, val_batch_size = 32, sub_t=1, slice_size=1, shuffle=True):
    file_name = os.path.join(file_name, 'wave2d')#actually wave easy
    train_data = WaveDataset(os.path.join(file_name, 'train.h5'), group = 'train', sub_t=sub_t, slice_size=slice_size)
    val_data = WaveDataset(os.path.join(file_name, 'val.h5'), group = 'val', sub_t=sub_t, slice_size=slice_size)
    test_data = WaveDataset(os.path.join(file_name, 'test.h5'), group = 'test', slice_size=slice_size)
    train_loader = DataLoader(train_data, batch_size= train_batch_size, shuffle=shuffle, num_workers=1, pin_memory=True,)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, pin_memory=True,)
    test_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False, pin_memory=True,)
    return train_loader, val_loader, test_loader


def load_rd(file_name, train_batch_size = 32, val_batch_size = 32, sub_t=1, slice_size=1, shuffle=True):
    file_name = os.path.join(file_name, 'rd')#actually wave easy
    train_data = ReacDiffDataset(os.path.join(file_name, 'train.h5'), group = 'train', sub_t=sub_t, slice_size=slice_size)
    val_data = ReacDiffDataset(os.path.join(file_name, 'val.h5'), group = 'val', sub_t=sub_t, slice_size=slice_size)
    test_data = ReacDiffDataset(os.path.join(file_name, 'test.h5'), group = 'test', sub_t=sub_t, slice_size=slice_size)
    train_loader = DataLoader(train_data, batch_size= train_batch_size, shuffle=shuffle, num_workers=1, pin_memory=True,)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, pin_memory=True,)
    test_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False, pin_memory=True,)
    return train_loader, val_loader, test_loader

def load_vort(file_name, train_batch_size = 32, val_batch_size = 32, sub_t=1, slice_size=1, shuffle=True, smooth=True):
    file_name = os.path.join(file_name, 'vorticity')
    train_data = VortDataset(os.path.join(file_name, 'train.h5'), group = 'train', slice_size=slice_size, smooth=smooth)
    val_data = VortDataset(os.path.join(file_name, 'val.h5'), group = 'val', slice_size=slice_size, smooth=smooth)
    test_data = VortDataset(os.path.join(file_name, 'test.h5'), group = 'test', slice_size=slice_size, smooth=smooth)
    train_loader = DataLoader(train_data, batch_size= train_batch_size, shuffle=shuffle, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader



def tokenize_dataset(token_dataset_path, run_name, train_loader, val_loader, test_loader, tokenizer, device):
    try:
        token_train = torch.load(f"{token_dataset_path}/{run_name}/train_token.pt")
        token_val = torch.load(f"{token_dataset_path}/{run_name}/val_token.pt")
        token_test = torch.load(f"{token_dataset_path}/{run_name}/test_token.pt")
    
    except: 
        token_train = []
        token_val = []
        token_test = []
        
        with torch.no_grad():
            for out_list, loader in zip([token_train, token_val, token_test], [train_loader, val_loader, test_loader]):
                for batch in loader:
                    sequences = batch.to(device).float()
                    t = sequences.shape[-1]

                    if batch.ndim==4:
                        sequences = rearrange(sequences, "b c h t-> (b t) c h")
                    elif batch.ndim==5:
                        sequences = rearrange(sequences, "b c h w t-> (b t) c h w")
                        
                    codes, indices = tokenizer(sequences, return_codes=True)

                    if batch.ndim==4:
                        indices = rearrange(indices, "(b t) c h -> b c h t", t=t)
                    elif batch.ndim==5:
                        indices = rearrange(indices, "(b t) c h w -> b c h w t", t=t)

                    out_list.append(indices.cpu().detach()) 

        token_train = torch.cat(token_train, axis=0)
        token_val = torch.cat(token_val, axis=0)
        token_test = torch.cat(token_test, axis=0)
        
        os.makedirs(f"{token_dataset_path}/{run_name}/", exist_ok=True)
        torch.save(token_train, f"{token_dataset_path}/{run_name}/train_token.pt")
        torch.save(token_val, f"{token_dataset_path}/{run_name}/val_token.pt")
        torch.save(token_test, f"{token_dataset_path}/{run_name}/test_token.pt")
    
    return token_train, token_val, token_test
