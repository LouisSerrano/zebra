import os
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, LlamaConfig
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
import einops
import sys
WORK_DIR = os.environ['WORK']
sys.path.append(f"{WORK_DIR}/zebra/")
from train.train_tokenizer_2d import TokenizerModel
from copy import deepcopy
import wandb
from functools import reduce
from torch.optim.lr_scheduler import CosineAnnealingLR

import random
from utils.load_data import get_data
from utils.load_data_new import load_wave, load_vort
#from trl import SFTTrainer
from time import time 


def collate_fn(batch, bos_token_id, eos_token_id, bot_token_id, eot_token_id,  max_length):
    concatenated = []
    for tokens in batch:
        concatenated.extend(tokens.tolist() + [special_token_id])
    concatenated = concatenated[:-1]  # Remove the last special token
    
    # Chunk concatenated list into max_length pieces
    chunks = [concatenated[i:i + max_length] for i in range(0, len(concatenated), max_length)]
    
    # Pad chunks to max_length
    padded_chunks = [chunk + [0] * (max_length - len(chunk)) for chunk in chunks]
    
    return torch.tensor(padded_chunks)

# Wrapper for collate_fn to pass additional arguments

class ZebraModel(pl.LightningModule):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.save_hyperparameters()

        config = LlamaConfig(**cfg.model)
        self.cfg = cfg
        self.model = LlamaForCausalLM(config)
        self.tokenizer = tokenizer #.cuda()
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
        self.sample_context = False
        self.accumulate_grad = 4
        self.custom_step=0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        sequences, context_sequences = batch
        sequences = rearrange(sequences, 'b c h w t -> b (t h w c)')
        context_sequences = rearrange(context_sequences, 'b k c h w t -> b k t (h w c)')

        opt = self.optimizers()
        sch = self.lr_schedulers()
        t = sequences.shape[-1]
        b = sequences.shape[0]
        k = context_sequences.shape[1]
        device = sequences.device

        cat_context_seq = []
        for j in range(b):
            context_seq = [self.bos_token_id]
            num_examples = random.randint(0, k)
            trajectory_size = t
            for ex in range(num_examples):
                if self.sample_context:
                    trajectory_size = random.randint(2, t)
                else:
                    trajectory_size = t
                context_seq.append(self.bot_token_id)
                context_seq.extend(rearrange(context_sequences[j, ex, :trajectory_size], 't h -> (t h)').tolist())
                context_seq.append(self.eot_token_id)

            context_seq.append(self.bot_token_id) # input_token_id
            context_seq.extend(sequences[j].tolist())
            context_seq.append(self.eot_token_id) # target_token_id
            context_seq.append(self.eos_token_id)
            cat_context_seq.append(context_seq)

        max_len=8192
        use_chunks=True
        if use_chunks:
            cat_context_seq = [item for sublist in cat_context_seq for item in sublist]
            chunks = [cat_context_seq[i:i + max_len] for i in range(0, len(cat_context_seq), max_len)]

            padded_chunks = []
            padded_labels = []
            for chunk in chunks:
                chunk = torch.tensor(chunk, device=device)
                if chunk.size(0) < max_len:
                    padding = torch.full((max_len - chunk.size(0),), self.pad_token_id, dtype=torch.long, device=device)
                    padding_label = torch.full((max_len - chunk.size(0),), -100, dtype=torch.long, device=device)
                    lbl = torch.cat([chunk, padding_label])
                    chunk = torch.cat([chunk, padding])
                    padded_chunks.append(chunk)
                    padded_labels.append(lbl)
                else:
                    padded_chunks.append(chunk)
                    padded_labels.append(chunk)
        else:
            max_len=max([len(seq) for seq in cat_context_seq])
            padded_chunks = []
            padded_labels = []
            for seq in cat_context_seq:
                seq = torch.tensor(seq, device=device)
                padding_length = max_len - len(seq)
                padding = torch.full((padding_length,), self.pad_token_id, dtype=torch.long, device=device)
                padding_label = torch.full((padding_length,), -100, dtype=torch.long, device=device)
                lbl = torch.cat([seq, padding_label])
                chunk = torch.cat([seq, padding])
                padded_chunks.append(chunk)
                padded_labels.append(lbl)
            padded_chunks = [torch.tensor(chunk) for chunk in padded_chunks]
            padded_labels = [torch.tensor(chunk) for chunk in padded_chunks]
        
     
        input_ids = torch.stack(padded_chunks)
        labels = torch.stack(padded_labels)
        print('input_ids', input_ids.shape)

        out = self.model(input_ids, labels=labels)
        loss = out.loss
        
        #opt.zero_grad()
        self.manual_backward(loss/self.accumulate_grad)

        if (self.custom_step + 1) % self.accumulate_grad == 0:
            opt.step()
            opt.zero_grad() 
            sch.step()
        
        self.custom_step+=1
        
        #opt.step()
        #sch.step()
        self.log('train_nll', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        sequences, context_sequences = batch
        sequences = rearrange(sequences, 'b c h w t -> b (t h w c)')
        context_sequences = rearrange(context_sequences, 'b k c h w t -> b k t (h w c)')

        opt = self.optimizers()
        sch = self.lr_schedulers()
        t = sequences.shape[-1]
        b = sequences.shape[0]
        k = context_sequences.shape[1]
        device = sequences.device

        cat_context_seq = []
        for j in range(b):
            context_seq = [self.bos_token_id]
            num_examples = random.randint(0, k)
            trajectory_size = t
            for ex in range(num_examples):
                if self.sample_context:
                    trajectory_size = random.randint(2, t)
                else:
                    trajectory_size = t
                context_seq.append(self.bot_token_id)
                context_seq.extend(rearrange(context_sequences[j, ex, :trajectory_size], 't h -> (t h)').tolist())
                context_seq.append(self.eot_token_id)

            context_seq.append(self.bot_token_id) # input_token_id
            #context_seq.extend(rearrange(sequences[j, :trajectory_size], 't h -> (t h)').tolist())
            context_seq.extend(sequences[j].tolist())
            context_seq.append(self.eot_token_id) # target_token_id
            context_seq.append(self.eos_token_id)
            cat_context_seq.append(context_seq)

        max_len=8192
        use_chunks=True
        if use_chunks:
            cat_context_seq = [item for sublist in cat_context_seq for item in sublist]
            chunks = [cat_context_seq[i:i + max_len] for i in range(0, len(cat_context_seq), max_len)]

            padded_chunks = []
            padded_labels = []
            for chunk in chunks:
                chunk = torch.tensor(chunk, device=device)
                if chunk.size(0) < max_len:
                    padding = torch.full((max_len - chunk.size(0),), self.pad_token_id, dtype=torch.long, device=device)
                    padding_label = torch.full((max_len - chunk.size(0),), -100, dtype=torch.long, device=device)
                    lbl = torch.cat([chunk, padding_label])
                    chunk = torch.cat([chunk, padding])
                    padded_chunks.append(chunk)
                    padded_labels.append(lbl)
                else:
                    padded_chunks.append(chunk)
                    padded_labels.append(chunk)
        else:
            max_len=max([len(seq) for seq in cat_context_seq])
            padded_chunks = []
            padded_labels = []
            for seq in cat_context_seq:
                seq = torch.tensor(seq, device=device)
                padding_length = max_len - len(seq)
                padding = torch.full((padding_length,), self.pad_token_id, dtype=torch.long, device=device)
                padding_label = torch.full((padding_length,), -100, dtype=torch.long, device=device)
                lbl = torch.cat([seq, padding_label])
                chunk = torch.cat([seq, padding])
                padded_chunks.append(chunk)
                padded_labels.append(lbl)
            padded_chunks = [torch.tensor(chunk) for chunk in padded_chunks]
            padded_labels = [torch.tensor(chunk) for chunk in padded_chunks]
    
        input_ids = torch.stack(padded_chunks)
        labels = torch.stack(padded_labels)
 
        out = self.model(input_ids, labels=labels)
        loss = out.loss

         #print('sequences', sequences.dtype)

        self.log('val_nll', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_loss', rel_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        #embedding_params = self.model.get_input_embeddings().weight
        #new_embedding_params = embedding_params[-256:]
        optimizer = AdamW([{"params": self.model.parameters()}], lr=self.cfg.train.learning_rate, weight_decay=1e-6)
        #optimizer = AdamW(self.model.parameters(), lr=self.cfg.train.learning_rate, weight_decay=1e-6)
        #scheduler = ConstantWithLinearDecayScheduler(optimizer, int(0.8*self.cfg.train.max_steps / 4), int(self.cfg.train.max_steps/4))
        #scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.train.max_steps//4, eta_min=1e-6)
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=100, num_training_steps=self.cfg.train.max_steps//4)
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
    def __init__(self, u, token, slice_size=20, num_context_images=5):
        self.u = u
        self.token = token 
        self.slice_size = slice_size
        self.num_context_images = num_context_images
        self.subsampling_t = 10
        self.random_t = False

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        #sub_t = random.choice([1, 2, 3]) # warning 25 does not work for all datasets.
        #sub_t = random.choice([1])
        #sub_t = self.subsampling_t
        # WARNING !!
        #sub_t = 1 # experiments with wave
        sub_t = 3
        images = self.u[idx, ..., ::sub_t] # self.subsampling_t
        tokens = self.token[idx, ..., ::sub_t]

        start_index = 0 

        min_index = (idx // 10)*10 # we have 10 trajectories per environment
        max_index = min_index + 9

        context_images = []
        context_tokens = []
        max_start_index = images.shape[-1] - self.slice_size # WARNING test ongoing
        ctx_idx_list = [i for i in range(min_index, max_index+1)] 
        ctx_idx_list.remove(idx)
        random.shuffle(ctx_idx_list)
        for j in range(self.num_context_images):
            if self.random_t:
                start_index = np.random.randint(0, max_start_index + 1)  # WARNING test ongoing
            ctx_idx = ctx_idx_list[j]
            context_images.append(self.u[ctx_idx, ..., ::sub_t][..., start_index:start_index + self.slice_size])
            context_tokens.append(self.token[ctx_idx, ..., ::sub_t][..., start_index:start_index + self.slice_size])

        context_images = torch.stack(context_images)
        context_tokens = torch.stack(context_tokens)

        return images, context_images, tokens, context_tokens

def cleanup():
    dist.destroy_process_group()

def display_device(model):
    device = next(model.parameters()).device
    if device.type == 'cuda':
        print("Model is on GPU")
    else:
        print("Model is on CPU")

@hydra.main(config_path="../config", config_name="llama_2d.yaml")
def main(cfg):
    
    dataset_name = cfg.dataset.dataset_name
    run_name = cfg.dataset.run_name

    #u_train, u_val, u_test = get_data(dataset_name)
    #"dashing-voice-876" #if dataset_name == "mp-pde-burgers-10" else "hardy-eon-661"
    batch_size=32
    if dataset_name=="wave2d":
        train_loader, val_loader, test_loader = load_wave("/lustre/fsn1/projects/rech/mdw/ueg82cz/", batch_size, batch_size, slice_size=30, shuffle=False)
    elif dataset_name=="vorticity":
        train_loader, val_loader, test_loader = load_vort("/lustre/fsn1/projects/rech/mdw/ueg82cz/", batch_size, batch_size, slice_size=30, shuffle=False) 

    vae_ckpt_path = f"{cfg.dataset.input_dir}/{dataset_name}/{run_name}.ckpt"
    tkn = TokenizerModel.load_from_checkpoint(vae_ckpt_path)
    tokenizer = tkn.model.float().eval()
    
    token_dataset_path = f"/lustre/fsn1/projects/rech/mdw/ueg82cz/{dataset_name}"
    
     
    token_train = []
    token_val = []
    token_test = []
    
    u_val = []
    
    with torch.no_grad():
        for out_list, loader in zip([token_val], [val_loader]):
            for batch in loader:
                sequences = batch.cuda()
                u_val.append(sequences.cpu())
                t = sequences.shape[-1]
                sequences = rearrange(sequences, "b c h w t-> (b t) c h w")
                codes, indices = tokenizer.tokenizer(sequences, return_codes=True)
                indices = rearrange(indices, "(b t) c h w -> b c h w t", t=t)
                out_list.append(indices.cpu().detach()) 

    #token_train = torch.cat(token_train, axis=0)
    token_val = torch.cat(token_val, axis=0)
    #token_test = torch.cat(token_test, axis=0)
    u_val = torch.cat(u_val, axis=0) 
        
    zebra_ckpt_path = cfg.dataset.checkpoint_path # WARNING to do in config 
    model = ZebraModel.load_from_checkpoint(zebra_ckpt_path)
        
    slice_size=cfg.dataset.slice_size
    num_context_images = cfg.dataset.num_context_images
    #trainset = TemporalDataset(token_train, slice_size=slice_size, num_context_images=num_context_images)
    testset = TemporalDataset(u_val, token_val, slice_size=slice_size, num_context_images=num_context_images)

    #run = wandb.init(project="zebra")
    #run_name = wandb.run.name
    #run.tags = (
    #        ("llama",)
    #        + (dataset_name,)
    #    )
    zebra_ckpt_dir = f"{cfg.dataset.output_dir}/{dataset_name}/{run_name}/"

    batch_size = 2

    #train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,  prefetch_factor=1)# collate_fn=tokenize_function)
    val_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=1)#, collate_fn=tokenize_function)
    #checkpoint_callback = ModelCheckpoint(dirpath=zebra_ckpt_dir, save_top_k=2, save_last=True, verbose=True, every_n_train_steps=1000, filename='{step}-{train_nll:.4f}', monitor="train_nll")
    #wandb_logger = WandbLogger(project="zebra", tags=[cfg.dataset.dataset_name, cfg.dataset.trainset, "token"])
    #trainer = pl.Trainer(max_steps=cfg.train.max_steps, devices="auto", accumulate_grad_batches=1, accelerator="gpu", logger=wandb_logger, callbacks=[checkpoint_callback], check_val_every_n_epoch=100, precision="16-mixed")
    #trainer.fit(model, train_loader, val_loader)
    
    start_time = time() 
    num_examples = 0 #random.randint(0, k)
    input_size = 1

    all_images = []
    all_context_images = []
    all_pred = []

    total_loss = 0
    cpt=0

    for images, context_images, tokens, context_tokens in val_loader:
        images = images.cuda()
        
        sequences, context_sequences = tokens, context_tokens 
        sequences = rearrange(sequences[..., :input_size], 'b c h w t -> b (t h w c)')
        context_sequences = rearrange(context_sequences, 'b k c h w t -> b k t (h w c)')

        t = sequences.shape[-1]
        b = sequences.shape[0]
        k = context_sequences.shape[1]
        device = sequences.device

        cat_context_seq = []
        for j in range(b):
            context_seq = [model.bos_token_id]
            trajectory_size = t
            for ex in range(num_examples):
                context_seq.append(model.bot_token_id)
                context_seq.extend(rearrange(context_sequences[j, ex, :trajectory_size], 't h -> (t h)').tolist())
                context_seq.append(model.eot_token_id)

            context_seq.append(model.bot_token_id) # input_token_id
            context_seq.extend(sequences[j].tolist())
            #context_seq.append(self.eot_token_id) # target_token_id
            #context_seq.append(self.eos_token_id)
            cat_context_seq.append(context_seq)
        
        input_ids = torch.tensor(cat_context_seq).cuda()
        rollout_len = 9
        output_len = 256*rollout_len
        max_length = output_len + input_ids.shape[1]
        with torch.no_grad():
            output = model.model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True, bad_words_ids=[[2048],[2049],[2050],[2051],[2052],[2053],[2054]])
        
        indices = rearrange(output[:, -output_len:], 'b (t h w c)  -> (b t h w) c' , t=rollout_len, h=16, w=16).cuda()
        quantized_pred = model.tokenizer.tokenizer.quantizers.get_output_from_indices(indices)
        quantized_pred = rearrange(quantized_pred, '(b t h w) c -> (b t) c h w', t=rollout_len, h=16, w=16)
        #pred = rearrange(pred, '(b h t) c -> (b t) c h', t=rollout_len, h=h_lat)
        
        with torch.no_grad(): 
            upred = model.tokenizer.tokenizer.decode(quantized_pred, cond=None)
        upred = rearrange(upred, '(b t) c h w -> b c h w t', t=rollout_len)
        
        loss = model.rel_loss(upred, images[..., 1:])
        print('loss', cpt, loss)
        total_loss += loss*batch_size 
        cpt+=batch_size
        all_images.append(np.array(images.cpu().detach()))
        all_context_images.append(np.array(context_images[:, :num_examples].cpu().detach()))
        all_pred.append(np.array(upred.cpu().detach()))
    
    all_images = np.concatenate(all_images)
    all_context_images = np.concatenate(all_context_images)
    all_pred = np.concatenate(all_pred)
    
    print('total_loss l2', total_loss/cpt, "time", time() - start_time)
    
    os.makedirs(f"/lustre/fsn1/projects/rech/mdw/ueg82cz/zebra/inference/{dataset_name}/", exist_ok=True)
    np.save(f"/lustre/fsn1/projects/rech/mdw/ueg82cz/zebra/inference/{dataset_name}/pred_{input_size}_{num_examples}.npy", all_pred) 
    np.save(f"/lustre/fsn1/projects/rech/mdw/ueg82cz/zebra/inference/{dataset_name}/gt_{input_size}_{num_examples}.npy", all_images)
    np.save(f"/lustre/fsn1/projects/rech/mdw/ueg82cz/zebra/inference/{dataset_name}/example_{input_size}_{num_examples}.npy", all_context_images)
         
        
        
 

if __name__ == "__main__":
    main()
