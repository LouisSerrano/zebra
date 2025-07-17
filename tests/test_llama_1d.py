import os
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertConfig, BertForMaskedLM
from torch.optim import AdamW
from einops import rearrange
#from basic1d_dataset import FNODatasetMultiple
import yaml
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
#from arom.utils.data.load_data import get_dynamics_data
import einops
import sys
sys.path.append('/home/serrano/Projects/zebra/')
from train.train_tokenizer_vqgan_mppde import TokenizerModel
from copy import deepcopy
#sys.path.append("/home/serrano/Projects/attentive-rom/")
#sys.path.append("/home/serrano/Projects/attentive-rom/baselines/")
#from baselines.test import BaselineModel
import wandb
import torch
import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding
import h5py
from functools import reduce
import time
from transformers import get_scheduler
from utils.load_data import get_data, get_data_ood
import pandas as pd
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, LlamaConfig
import matplotlib.pyplot as plt


# Wrapper for collate_fn to pass additional arguments

class ZebraModel(pl.LightningModule):
    def __init__(self, tokenizer, cfg):
        super().__init__()

        config = LlamaConfig(**cfg.model)
        self.cfg = cfg
        self.model = LlamaForCausalLM(config).float()
        self.tokenizer = tokenizer.cuda().float()
        #self.codebook = self.tokenizer.tokenizer.quantizers._codebook.embed.squeeze()
        self.automatic_optimization = False
        self.rel_loss = RelativeL2()
 
        self.bos_token_id = cfg.model.bos_token_id
        self.eos_token_id = cfg.model.eos_token_id
        self.context_token_id = cfg.model.context_token_id
        self.input_token_id = cfg.model.input_token_id
        self.target_token_id = cfg.model.target_token_id
        self.bot_token_id = cfg.model.bot_token_id
        self.eot_token_id = cfg.model.eot_token_id
        self.pad_token_id= cfg.model.pad_token_id
        self.target_size = self.cfg.dataset.slice_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        images, context_images = batch

        opt = self.optimizers()
        sch = self.lr_schedulers()
        h_og = images.shape[1]
        t = images.shape[-1]
        b = images.shape[0]
        device = images.device

        with torch.no_grad():
            x = rearrange(images, "b h c t -> (b t) c h").float()
            codes, indices = self.tokenizer.tokenizer(x, return_codes=True)
            h_lat = indices.shape[1]
            sequences = rearrange(indices, '(b t) h c -> b (t h c)', t=t) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}

        with torch.no_grad():
            k = context_images.shape[1]
            x = rearrange(context_images, "b k h c t -> (b k t) c h").float()
            codes, indices = self.tokenizer.tokenizer(x, return_codes=True)
            h_lat = indices.shape[1]
            context_sequences = rearrange(indices, '(b k t) h c -> b k (t h c)', t=t, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}

        inp = torch.cat([torch.full((b,1), self.input_token_id, device=device), sequences[..., :32]], axis=1)
        target = torch.cat([torch.full((b,1), self.target_token_id, device=device), sequences[..., 32:]], axis=1)
        context = torch.cat([torch.full((b, k, 1), self.bot_token_id, device=device), context_sequences, torch.full((b, k, 1), self.eot_token_id, device=device)], axis=2)
        context = rearrange(context, 'b k t-> b (k t)')
        input_ids = torch.cat([torch.full((b,1), self.bos_token_id, device=device), context, inp, target, torch.full((b,1), self.eos_token_id, device=device),], axis=1)
        
        labels=input_ids.clone()
        out = self.model(input_ids)

        output_len = 32 * (self.target_size - 1) + 1
        logits = out.logits[:, -output_len:]
        labels = labels[:, -output_len:]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        sch.step()
        self.log('train_nll', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        images, context_images = batch
        t = images.shape[-1]
        b = images.shape[0]
        device = images.device
        with torch.no_grad():
            #print('batch', batch.shape)
            x = rearrange(images, "b h c t -> (b t) c h").float()
            #x = rearrange(batch, "b h c t -> b c t h").float()
            codes, indices = self.tokenizer.tokenizer(x, return_codes=True)
            #print('indices', indices)
            #indices = rearrange(indices, "(b t) h c -> b c t h").float()
            h_lat = indices.shape[1]
            sequences = rearrange(indices, '(b t) h c -> b (t h c)', t=t) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}

        with torch.no_grad():
            k = context_images.shape[1]
            x = rearrange(context_images, "b k h c t -> (b k t) c h").float()
            codes, indices = self.tokenizer.tokenizer(x, return_codes=True)
            h_lat = indices.shape[1]
            context_sequences = rearrange(indices, '(b k t) h c -> b k (t h c)', t=t, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}
        
        inp = torch.cat([torch.full((b,1), self.input_token_id, device=device), sequences[..., :32]], axis=1)
        target = torch.cat([torch.full((b,1), self.target_token_id, device=device), sequences[..., 32:]], axis=1)
        context = torch.cat([torch.full((b, k, 1), self.bot_token_id, device=device), context_sequences, torch.full((b, k, 1), self.eot_token_id, device=device)], axis=2)
        context = rearrange(context, 'b k t-> b (k t)')
        input_ids = torch.cat([torch.full((b,1), self.bos_token_id, device=device), context, inp, target, torch.full((b,1), self.eos_token_id, device=device),], axis=1)

        shift_labels = x
        patch_size = sequences.shape[-1]

        labels = input_ids.clone()
        out = self.model(input_ids)
        output_len =  32 * (self.target_size - 1) + 1
        logits = out.logits[:, -output_len:]
        labels = labels[:, -output_len:]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        self.log('val_nll', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW([{"params": self.model.parameters()}], lr=self.cfg.train.learning_rate, weight_decay=1e-6)
        scheduler = ConstantWithLinearDecayScheduler(optimizer, int(0.8*self.cfg.train.max_steps), self.cfg.train.max_steps)
        return [optimizer], [scheduler]


class RelativeL2(nn.Module):
    def forward(self, x, y):
        x = rearrange(x, "b ... -> b (...)")
        y = rearrange(y, "b ... -> b (...)")
        diff_norms = torch.linalg.norm(x - y, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        return (diff_norms / y_norms).mean()

def decode_from_indices(ids, image_size=256):
    b = ids.shape[0]
    t = ids.shape[1]
    ids = ids.view(b, t, image_size, 1)
    return ids


class ConstantWithLinearDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, constant_steps, total_steps, last_epoch=-1):
        """
        Initializes the scheduler.

        Parameters:
        - optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.
        - constant_steps (int): Number of steps to keep the learning rate constant.
        - total_steps (int): Total number of steps for training.
        - last_epoch (int, optional): The index of the last epoch. Default: -1.
        """
        self.constant_steps = constant_steps
        self.total_steps = total_steps
        super(ConstantWithLinearDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.constant_steps:
            # Constant learning rate
            return self.base_lrs
        else:
            # Linearly decay the learning rate
            decay_steps = self.total_steps - self.constant_steps
            decay_factor = (self.total_steps - self.last_epoch) / decay_steps
            return [base_lr * decay_factor for base_lr in self.base_lrs]

def extract_overlapping_patches_1d(array, patch_size, overlap):
    """
    Extract overlapping patches from a 1D array.

    Parameters:
    - array (numpy.ndarray): The input 1D array from which to extract patches.
    - patch_size (int): The size of each patch.
    - overlap (int): The number of elements by which patches overlap.

    Returns:
    - patches (numpy.ndarray): Array of extracted patches.
    """
    stride = patch_size - overlap
    n_patches = (len(array) - patch_size) // stride + 1
    patches = torch.stack([array[i * stride:i * stride + patch_size] for i in range(n_patches)])
    return patches


class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, u, slice_size=20, subsampling_t=10, num_context_images=5, context_slice_size=10):
        self.u = u 
        self.slice_size = slice_size
        self.context_slice_size= context_slice_size
        self.num_context_images = num_context_images
        self.subsampling_t = subsampling_t

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        images = self.u[idx, ..., ::self.subsampling_t]

        max_start_index = images.shape[-1] - self.slice_size
        if max_start_index < 0:
            raise ValueError("Slice size is larger than the sequence length.")
        #start_index = np.random.randint(0, max_start_index + 1)
        start_index = 0
        images = images[..., start_index:start_index + self.slice_size]

        min_index = (idx // 10)*10 # we have 10 trajectories per environment
        max_index = min_index + 9

        context_images = []
        ctx_idx_list = [i for i in range(min_index, max_index+1)] 
        ctx_idx_list.remove(idx)
        random.shuffle(ctx_idx_list)
        for j in range(self.num_context_images):
            ctx_idx = ctx_idx_list[j]
            context_images.append(self.u[ctx_idx, ..., ::self.subsampling_t][..., start_index:start_index + self.context_slice_size])

        context_images = torch.stack(context_images)

        return images.float(), context_images.float()
    

def get_test_metrics(loader, model, num_token=2, inp_size=1, num_examples=0, include_special_token=False):
    my_loss = RelativeL2()
    count = 0
    ntest = 0
    l2_loss = 0
    nll_loss = 0

    with torch.no_grad():
        
        for batch in loader:
            count+=1                
            images, context_images = batch
            images = images.cuda()
            context_images = context_images.cuda()

            # test with this
            #context_images = context_images[..., :]
            #images = images[..., #1:-1] 
            #images = images[..., :-1] there was this bug for some reason
        
            t = images.shape[-1]
            t_context = context_images.shape[-1]
            b = images.shape[0]
            device = images.device
            with torch.no_grad():
                x = rearrange(images, "b h c t -> (b t) c h").float()
                codes, indices = model.tokenizer.tokenizer(x, return_codes=True)
                h_lat = indices.shape[1]
                sequences = rearrange(indices, '(b t) h c -> b (t h c)', t=t) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}

            with torch.no_grad():
                k = context_images.shape[1]
                x = rearrange(context_images, "b k h c t -> (b k t) c h").float()
                codes, indices = model.tokenizer.tokenizer(x, return_codes=True)
                h_lat = indices.shape[1]
                context_sequences = rearrange(indices, '(b k t) h c -> b k (t h c)', t=t_context, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}


            inp = torch.cat([torch.full((b,1), model.bot_token_id, device=device), sequences[..., :num_token*inp_size*16]], axis=1)
            if num_examples > 0:
                context = torch.cat([torch.full((b, k, 1), model.bot_token_id, device=device), context_sequences, torch.full((b, k, 1), model.eot_token_id, device=device)], axis=2)
                context = rearrange(context[:, :num_examples], 'b k t-> b (k t)')
                if include_special_token:
                    input_ids = torch.cat([torch.full((b,1), model.bos_token_id, device=device),
                                           torch.full((b,1), model.context_token_id, device=device),
                                           context,
                                           torch.full((b,1), model.target_token_id, device=device),
                                            inp], axis=1)
                else:    
                    input_ids = torch.cat([torch.full((b,1), model.bos_token_id, device=device),
                                        context, inp], axis=1)
            else:
                if include_special_token:
                    input_ids = torch.cat([torch.full((b,1), model.bos_token_id, device=device), 
                                           torch.full((b,1), model.target_token_id, device=device),
                                           inp], axis=1)
                else:
                    input_ids = torch.cat([torch.full((b,1), model.bos_token_id, device=device), inp], axis=1)

            rollout_len = t - inp_size
            h_lat = 16
            output_len = rollout_len*h_lat*num_token + 1
            max_length = input_ids.shape[1] + output_len
            output = model.model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=0.1, bad_words_ids=[[256],[257],[258],[259],[260],[261],[262]]) #do_sample=True, temperature=0.4, top_k=3)
            indices = rearrange(output[:, -output_len:-1], 'b (t h c) -> (b h t) c', t=rollout_len, h=h_lat).cuda()
            indices = torch.clamp(indices, 0, 255)
            #print(indices.max())
            quantized_pred = model.tokenizer.tokenizer.quantizers.get_output_from_indices(indices)
            quantized_pred = rearrange(quantized_pred, '(b h t) c -> (b t) c h', t=rollout_len, h=h_lat)
            
            upred = model.tokenizer.tokenizer.decode(quantized_pred, cond=None)

            
            upred = rearrange(upred, '(b t) c h -> b h c t', t=rollout_len)

            l2_loss += my_loss(upred, images[..., inp_size:])*b
            ntest += b        

        l2_loss = l2_loss / ntest

    return l2_loss.item(), upred.cpu().detach(), images.cpu(), context_images.cpu()


def plot_no_context(images, upred, idx, rollout_len=8, inp_size=2, dataset_name="advection", sub_t=10):
    root_dir = "/home/serrano/Projects/zebra/test_zebra/plots_ood"
    # Set up the figure with subplots, arranging 3 plots in the first row and 2 in the second
    fig, axs = plt.subplots(1, 3, figsize=(15, 6)) #gridspec_kw={'height_ratios': [1], 'width_ratios': [1, 1, 1]})

    # Set the title for the entire figure
    fig.suptitle('Prompt = Initial condition', fontsize=16, y=1.03)

    # Flatten the axis array for easier indexing (3+2=5 total plots)
    axs = axs.flatten()

    for t in range(inp_size):
        axs[0].plot(images[idx].squeeze().cpu().detach()[:, t], label=f'Time {t}')
    axs[0].set_title(f'Initial conditions', fontsize=12)
    axs[0].set_ylabel('Value', fontsize=10)
    axs[0].grid(True)
    axs[0].legend(loc='upper right', fontsize=6)

    for t in range(0, 10):
        axs[1].plot(images[idx].squeeze().cpu().detach()[:, t], label=f'Time {t}')
    axs[1].set_title(f'Ground Truth', fontsize=12)
    axs[1].set_ylabel('Value', fontsize=10)
    axs[1].grid(True)
    axs[1].legend(loc='upper right', fontsize=6)

    for t in range(inp_size):
        axs[2].plot(images[idx].squeeze().cpu().detach()[:, t], label=f'Time {t}')
    for t in range(rollout_len):
        axs[2].plot(upred[idx].squeeze().cpu().detach()[:, t], label=f'Time {t+inp_size}')
    axs[2].set_title(f'Generation', fontsize=12)
    axs[2].set_ylabel('Value', fontsize=10)
    axs[2].grid(True)
    axs[2].legend(loc='upper right', fontsize=6)

    plt.tight_layout()

    # Show the plot
    os.makedirs(f"{root_dir}/{dataset_name}/zeroshot/", exist_ok=True)
    plt.savefig(f"{root_dir}/{dataset_name}/zeroshot/inp_{inp_size}_subt_{sub_t}_idx_{idx}.pdf")
    plt.close()

def plot_context_1(images, context_images, upred, idx, rollout_len=9, inp_size=1, dataset_name="advection", sub_t=10):
    root_dir = "/home/serrano/Projects/zebra/test_zebra/plots_ood"
    # Set up the figure with subplots, arranging 3 plots in the first row and 2 in the second
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})

    # Set the title for the entire figure
    fig.suptitle('Prompt = Context Images + initial condition', fontsize=16, y=1.03)

    # Flatten the axis array for easier indexing (3+2=5 total plots)
    axs = axs.flatten()

    # Loop over the examples and time steps
    for ex in range(1):
        for t in range(10):  # Simplified range
            axs[ex].plot(context_images[idx, ex].squeeze().cpu().detach()[:, t], label=f'Time {t}')
        
        # Add grid, labels, and legend to each subplot for better readability
        axs[ex].set_title(f'Example {ex + 1}', fontsize=12)
        axs[ex].set_ylabel('Value', fontsize=10)
        axs[ex].grid(True)
        axs[ex].legend(loc='upper right', fontsize=6)

    for t in range(inp_size):
        axs[1].plot(images[idx].squeeze().cpu().detach()[:, t], label=f'Time {t}')
    axs[1].set_title(f'Initial condition', fontsize=12)
    axs[1].set_ylabel('Value', fontsize=10)
    axs[1].grid(True)
    axs[1].legend(loc='upper right', fontsize=6)

    for t in range(0, 10):
        axs[2].plot(images[idx].squeeze().cpu().detach()[:, t], label=f'Time {t}')
    axs[2].set_title(f'Ground Truth', fontsize=12)
    axs[2].set_ylabel('Value', fontsize=10)
    axs[2].grid(True)
    axs[2].legend(loc='upper right', fontsize=6)

    for t in range(inp_size):
        axs[3].plot(images[idx].squeeze().cpu().detach()[:, t], label=f'Time {t}')
    for t in range(rollout_len):
        axs[3].plot(upred[idx].squeeze().cpu().detach()[:, t], label=f'Time {t+inp_size}')
    axs[3].set_title(f'Generation', fontsize=12)
    axs[3].set_ylabel('Value', fontsize=10)
    axs[3].grid(True)
    axs[3].legend(loc='upper right', fontsize=6)

    # Adjust layout to prevent overlapping of titles and labels
    plt.tight_layout()

    # Show the plot
    os.makedirs(f"{root_dir}/{dataset_name}/oneshot/", exist_ok=True)
    plt.savefig(f"{root_dir}/{dataset_name}/oneshot/inp_{inp_size}_subt_{sub_t}_idx_{idx}.pdf")
    plt.close()


def plot_context_5(images, context_images, upred, idx, rollout_len=9, inp_size=1, dataset_name="advection", sub_t=10):
    root_dir = "/home/serrano/Projects/zebra/test_zebra/plots_ood"
    # Set up the figure with subplots, arranging 3 plots in the first row and 2 in the second
        
    # Set up the figure with subplots, arranging 3 plots in the first row and 2 in the second
    fig, axs = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'height_ratios': [1, 1, 1], 'width_ratios': [1, 1, 1]})

    # Set the title for the entire figure
    fig.suptitle('Prompt = Context Images + initial condition', fontsize=16, y=1.03)

    # Flatten the axis array for easier indexing (3+2=5 total plots)
    axs = axs.flatten()

    # Loop over the examples and time steps
    for ex in range(5):
        for t in range(10):  # Simplified range
            axs[ex].plot(context_images[idx, ex].squeeze().cpu().detach()[:, t], label=f'Time {t}')
        
        # Add grid, labels, and legend to each subplot for better readability
        axs[ex].set_title(f'Example {ex + 1}', fontsize=12)
        axs[ex].set_ylabel('Value', fontsize=10)
        axs[ex].grid(True)
        axs[ex].legend(loc='upper right', fontsize=6)

    axs[5].plot(images[idx].squeeze().cpu().detach()[:, 0], label=f'Time {0}')
    axs[5].set_title(f'Initial condition', fontsize=12)
    axs[5].set_ylabel('Value', fontsize=10)
    axs[5].grid(True)
    axs[5].legend(loc='upper right', fontsize=6)

    for t in range(0, 10):
        axs[6].plot(images[idx].squeeze().cpu().detach()[:, t], label=f'Time {t}')
    axs[6].set_title(f'Ground Truth', fontsize=12)
    axs[6].set_ylabel('Value', fontsize=10)
    axs[6].grid(True)
    axs[6].legend(loc='upper right', fontsize=6)

    for t in range(inp_size):
        axs[7].plot(images[idx].squeeze().cpu().detach()[:, t], label=f'Time {t}')
    for t in range(rollout_len):
        axs[7].plot(upred[idx].squeeze().cpu().detach()[:, t], label=f'Time {t+inp_size}')

    axs[7].set_title(f'Generation', fontsize=12)
    axs[7].set_ylabel('Value', fontsize=10)
    axs[7].grid(True)
    axs[7].legend(loc='upper right', fontsize=6)

    # Hide the last unused subplot if there are only 5 total plots
    axs[-1].axis('off')

    # Adjust layout to prevent overlapping of titles and labels
    plt.tight_layout()

    # Show the plot
    os.makedirs(f"{root_dir}/{dataset_name}/fiveshot/", exist_ok=True)
    plt.savefig(f"{root_dir}/{dataset_name}/fiveshot/inp_{inp_size}_subt_{sub_t}_idx_{idx}.pdf")
    plt.close()


@hydra.main(config_path="../config", config_name="llama_burgers.yaml")
def main(cfg):
    
    dataset_name = cfg.dataset.dataset_name
    run_name_dict = {"combined_equation": "misunderstood-armadillo-2084", #"youthful-water-2093", # "rural-dust-1612",#-> sub_t in 1, 2, 5, 10 #"glad-pine-1611", #sub_t=10
                    "wave_varying_boundary": "lively-capybara-1714", #"resilient-bush-1255", #"trim-dawn-1250", #"sunny-yogurt-1183",
                    "wave_fixed_boundary": "gentle-dragon-1701", #"crimson-lion-1182",
                    "heat2": "genial-bush-1685", #"dark-brook-2139", #"iconic-wave-2128", #"quiet-glade-2125", #"faithful-galaxy-2117", #"hopeful-dew-2120", #"smart-oath-2114", #"genial-bush-1685", #"vivid-snowball-1170",
                    "advection": "gentle-firebrand-1973", #"dashing-puddle-1700", #"electric-oath-1168",
                    "burgers_nu_forcing2":"spring-planet-1689",}# "effortless-glitter-1248"} #sweet-dream-1249 #fine-lion-1244

    run_name = run_name_dict[dataset_name]
    u_train, u_val, u_test = get_data(dataset_name)
    include_special_token=False
    num_token = 2

    if dataset_name == "wave_varying_boundary":
        sub_t=4
        num_context = 3
        slice_size = 15#+1

    elif dataset_name == "wave_fixed_boundary":
        sub_t=1
        num_context = 3
        slice_size = 10#+1

    else:
        sub_t=10
        num_context = 5
        slice_size = 10#+1

    if dataset_name == 'combined_equation':
        include_special_token = True
        num_token = 2 # 1 before 

    trainset = TemporalDataset(torch.from_numpy(u_train).float(), slice_size=slice_size, subsampling_t=sub_t, num_context_images=num_context, context_slice_size=slice_size)
    testset = TemporalDataset(torch.from_numpy(u_test).float(), slice_size=slice_size, subsampling_t=sub_t, num_context_images=num_context, context_slice_size=slice_size)

    #wandb.init(project="zebra")
    #run_name = wandb.run.name

    val_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=10, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=1)#, collate_fn=tokenize_function)

    zebra_ckpt_path = f"/data/serrano/zebra_results/zebra_patch/{dataset_name}/{run_name}/last.ckpt"
    model = ZebraModel.load_from_checkpoint(zebra_ckpt_path).cuda().eval()
    tokenizer = model.tokenizer.tokenizer.float().eval()

    num_max_frames = 3+1
    num_max_examples = num_context+1 #1

    # Create an empty DataFrame with columns as number of examples and rows as number of frames
    df = pd.DataFrame(index=range(num_max_frames), columns=range(num_max_examples))

    # Fill the DataFrame with values using a for loop (some values will be missing)
    l2_loss, pred, images, context_images = get_test_metrics(val_loader, model, num_token=num_token, inp_size=1, num_examples=1, include_special_token=include_special_token)
    print(f"dataset: {dataset_name}, num_examples: {1}, num_frames: {1}, loss: {l2_loss}")

if __name__ == "__main__":
    main()
