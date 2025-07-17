"""
Finetuning script for 1D Llama-Unet model.
"""
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
from train.train_tokenizer_vqgan_mppde import TokenizerModel
from copy import deepcopy
import wandb
from functools import reduce
import random
from utils.load_data import get_data
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.init as init


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
        self.sample_context = False
        self.weight_nll = 0
        self.gradient_accumulation = 5 #5
        self.noise_level = 5e-4

    def set_no_decoder(self):
        num_codes=1  #16
        self.num_codes = num_codes
        self.context_token = nn.Parameter(torch.randn(num_codes, self.cfg.model.hidden_size).to(self.model.device))
        padding = "zeros" if self.cfg.dataset.dataset_name in ['wave_varying_boundary', 'wave_fixed_boundary'] else 'circular'

        self.neural_operator = Unet(n_input_scalar_components=1,
                                    n_input_vector_components=0,
                                    n_output_scalar_components=1,
                                    n_output_vector_components=0,
                                    time_history=1,
                                    time_future=1,
                                    hidden_channels=64,
                                    activation="gelu",
                                    norm=True,
                                    ch_mults= (1, 1, 2, 2),
                                    is_attn=(False, False, False, False),
                                    mid_attn = False,
                                    n_blocks= 1,
                                    param_conditioning = None,
                                    use_scale_shift_norm = False,
                                    use1x1 = False,
                                    n_dims = 1,
                                    code_dim=self.cfg.model.hidden_size).to(self.model.device)
       

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
            k = context_images.shape[1]
            x = rearrange(context_images, "b k h c t -> (b k t) c h").float()
            codes, indices = self.tokenizer.tokenizer(x, return_codes=True)
            h_lat = indices.shape[1]
            context_sequences = rearrange(indices, '(b k t) h c -> b k t (h c)', t=t, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}

        cat_context_seq = []
        num_examples = random.randint(1, k) # for simplification and with context

        for j in range(b):
            context_seq = [self.bos_token_id]
            trajectory_size = t
            for ex in range(num_examples):
                if self.sample_context:
                    trajectory_size = random.randint(2, t)
                else:
                    trajectory_size = t
                context_seq.append(self.bot_token_id)
                context_seq.extend(rearrange(context_sequences[j, ex, :trajectory_size], 't h -> (t h)').tolist())
                context_seq.append(self.eot_token_id)
            context_seq.append(self.eos_token_id)
            #context_seq.append(self.context_token_id) # new token that will be used for conditionning the fno

            cat_context_seq.append(context_seq)

        max_len=2048
        use_chunks=False
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

        t = images.shape[-1]

        #out = self.model(input_ids, labels=labels, output_hidden_states=True)
        #embed = self.model.model.model.embed_tokens(input_ids)
        embed = self.model.model.embed_tokens(input_ids)
        embed = torch.cat([embed, self.context_token[None, ...].repeat(b,1,1)], axis=1)
        out = self.model(inputs_embeds=embed, output_hidden_states=True)
        emb = out.hidden_states[-1][:, -self.num_codes:, :]  # (b h) we take the hidden state from the last layer and and the last token (context token)
        #emb = out.hidden_states[-1].max(1)[0] 
        #emb = self.mlp(emb)
        nll = 0#out.loss
        #emb = out.hidden_states[-1][:, -1, :]  # (b h) we take the hidden state from the last layer and the last token (context token)
        #emb = self.mlp(emb)
        #emb = out.hidden_states[-1].max(1)[0] # (b h) we take the hidden state from the last layer and do a max pool
        #emb = emb[:, None, :]
        #emb = emb.repeat(t, 1, 1) # b h -> b t h  
        emb = einops.repeat(emb, 'b h c -> (b t) h c', t=t)
        #emb = rearrange(emb, 'b t h -> (b t) h')
        x = rearrange(images, 'b h c t -> (b t) c h')
        noise = torch.randn_like(x)*self.noise_level
        pred = self.neural_operator(x+noise, z=emb)
        pred = rearrange(pred, '(b t) c h -> b h c t', t=t)
        loss = self.rel_loss(pred[..., :-1], images[..., 1:]).mean()

        #loss = out.loss
        
        #opt.zero_grad()
        
        self.manual_backward(loss / self.gradient_accumulation)
        if (batch_idx + 1)%self.gradient_accumulation==0:
            opt.step()
            opt.zero_grad()
            sch.step()

        self.log('train_relative_l2', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_nll', nll, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_loss', rel_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        images, context_images = batch
       
        #print('batch', batch.dtype)
        t = images.shape[-1]
        b = images.shape[0]
        device = images.device
        
        with torch.no_grad():
            k = context_images.shape[1]
            x = rearrange(context_images, "b k h c t -> (b k t) c h").float()
            codes, indices = self.tokenizer.tokenizer(x, return_codes=True)
            h_lat = indices.shape[1]
            context_sequences = rearrange(indices, '(b k t) h c -> b k t (h c)', t=t, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}

        cat_context_seq = []
        num_examples = random.randint(1, k) # for simplification and with context

        for j in range(b):
            context_seq = [self.bos_token_id]
            trajectory_size = t
            for ex in range(num_examples):
                if self.sample_context:
                    trajectory_size = random.randint(2, t)
                else:
                    trajectory_size = t
                context_seq.append(self.bot_token_id)
                context_seq.extend(rearrange(context_sequences[j, ex, :trajectory_size], 't h -> (t h)').tolist())
                context_seq.append(self.eot_token_id)
            context_seq.append(self.eos_token_id)
            #context_seq.append(self.context_token_id) 
            #context_seq.append(self.context_token_id) # new token that will be used for conditionning the fno

            cat_context_seq.append(context_seq)

        max_len=2048
        use_chunks=False
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

        t = images.shape[-1]

        #out = self.model(input_ids, labels=labels, output_hidden_states=True)
        #embed = self.model.model.model.embed_tokens(input_ids)
        embed = self.model.model.embed_tokens(input_ids)
        embed = torch.cat([embed, self.context_token[None, ...].repeat(b,1,1)], axis=1)
        out = self.model(inputs_embeds=embed, output_hidden_states=True)
        emb = out.hidden_states[-1][:, -self.num_codes:, :]  # (b h) we take the hidden state from the last layer and and the last token (context token)
        #emb = out.hidden_states[-1].max(1)[0] 
        #emb = self.mlp(emb)
        #emb = self.mlp(emb) #
        #emb = emb[:, None, :].repeat(1, t, 1) # b h -> b t h

        x = images[..., 0]
        x = rearrange(x, 'b h c -> b c h')
        u_pred = []
        for time in range(t-1):
            pred = self.neural_operator(x, z=emb)
            x = pred.detach()
            u_pred.append(x)

        pred = torch.stack(u_pred, axis=-1)
        pred = rearrange(pred, 'b c h t -> b h c t')
        #emb = rearrange(emb, 'b t h -> (b t) h')
        #x = rearrange(images, 'b h c t -> (b t) c h')
        #pred = self.neural_operator(x, emb)
        #pred = rearrange(pred, '(b t) c h -> b h c t', t=t)
        loss = self.rel_loss(pred, images[..., 1:]).mean()

        self.log('val_relatve_l2', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_nll', 0, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_loss', rel_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        #embedding_params = self.model.get_input_embeddings().weight
        #new_embedding_params = embedding_params[-256:]
        #self.model.model.embed
        self.model.model.embed_tokens.requires_grad_(False)
        optimizer = AdamW([{"params": self.model.parameters(), "lr":3e-4},
                            {"params": self.neural_operator.parameters(), "lr":5e-4},
                            {"params": self.context_token, 'lr': 3e-4},],
                           #{"params": self.mlp.parameters(), "lr":5e-4}],
                            lr=1e-3, weight_decay=1e-4) #{"params": self.mlp.parameters(), "lr":1e-3}],
        #optimizer = AdamW(self.model.parameters(), lr=self.cfg.train.learning_rate, weight_decay=1e-6)
        #scheduler = ConstantWithLinearDecayScheduler(optimizer, int(0.8*self.cfg.train.max_steps/self.gradient_accumulation), int(self.cfg.train.max_steps/self.gradient_accumulation))
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= int(self.cfg.train.max_steps//6), gamma=0.5) #5
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=1000, num_training_steps=int(self.cfg.train.max_steps/self.gradient_accumulation)) # 100 warmup steps

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
    def __init__(self, u, slice_size=20, num_context_images=5, sub_t=[10]):
        self.u = u 
        self.slice_size = slice_size
        self.num_context_images = num_context_images
        self.subsampling_t = sub_t
        print('self.subsampling_t', self.subsampling_t)
        self.random_t = False

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        #sub_t = random.choice([1, 2, 5, 10, 25]) # heat, burgers
        #sub_t = random.choice([1, 2, 4, 5, 7, 10, 14]) # advection, combined_equation
        #sub_t = 10
        #sub_t = 1 #wave celerity
        #sub_t = random.choice([2, 4])  #wave boundary

        #sub_t = random.choice([1])
        #sub_t = self.subsampling_t
        # WARNING !!
        #sub_t = 1 # experiments with wave
        sub_t= random.choice(self.subsampling_t)
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
        for j in range(self.num_context_images):
            if self.random_t:
                start_index = np.random.randint(0, max_start_index + 1)  # WARNING test ongoing
            ctx_idx = ctx_idx_list[j]
            context_images.append(self.u[ctx_idx, ..., ::sub_t][..., start_index:start_index + self.slice_size])

        context_images = torch.stack(context_images)

        return images.float(), context_images.float()

def cleanup():
    dist.destroy_process_group()

def display_device(model):
    device = next(model.parameters()).device
    if device.type == 'cuda':
        print("Model is on GPU")
    else:
        print("Model is on CPU")

@hydra.main(config_path="../config", config_name="llama_finetune_fno.yaml")
def main(cfg):
    torch.set_default_dtype(torch.float32)
    print('cfg', cfg)
    dataset_name = cfg.dataset.dataset_name
    run_name = cfg.dataset.run_name

    u_train, u_val, u_test = get_data(dataset_name)

    #"dashing-voice-876" #if dataset_name == "mp-pde-burgers-10" else "hardy-eon-661"

    zebra_ckpt_path = f"{cfg.dataset.input_dir}/{dataset_name}/{run_name}/last.ckpt"
    #tkn = TokenizerModel.load_from_checkpoint(vae_ckpt_path)
    #tokenizer = tkn.model.float().eval()

    #print(u_train.shape)
    model = ZebraModel.load_from_checkpoint(zebra_ckpt_path).cuda()
    model.set_no_decoder()

    slice_size=cfg.dataset.slice_size
    num_context_images = cfg.dataset.num_context_images

    if dataset_name == "wave_varying_boundary":
        sub_t=4
        num_context_images = 3
        slice_size = 15#+1

    else:
        num_context_images = 5
        slice_size = 10
        sub_t=10
    
    trainset = TemporalDataset(torch.from_numpy(u_train).float(), slice_size=slice_size, num_context_images=num_context_images, sub_t=[sub_t])
    testset = TemporalDataset(torch.from_numpy(u_val).float(), slice_size=slice_size, num_context_images=num_context_images, sub_t=[sub_t])

    run = wandb.init(project="zebra")
    run_name = wandb.run.name
    run.tags = (
            ("llama_fno",)
            + (dataset_name,)
        )
    zebra_ckpt_dir = f"{cfg.dataset.output_dir}/{dataset_name}/{run_name}/"

    batch_size = 4

    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,  prefetch_factor=1)# collate_fn=tokenize_function)
    val_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=1)#, collate_fn=tokenize_function)
    checkpoint_callback = ModelCheckpoint(dirpath=zebra_ckpt_dir, save_top_k=2, save_last=True, verbose=True, every_n_train_steps=1000, filename='{step}-{train_nll:.4f}', monitor="train_relative_l2")
    wandb_logger = WandbLogger(project="zebra", tags=[cfg.dataset.dataset_name, cfg.dataset.trainset, "token"])
    trainer = pl.Trainer(max_steps=cfg.train.max_steps, devices="auto", accumulate_grad_batches=cfg.train.gradient_accumulation_steps, accelerator="gpu", logger=wandb_logger, callbacks=[checkpoint_callback], check_val_every_n_epoch=10)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
