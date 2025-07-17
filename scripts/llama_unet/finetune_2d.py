"""
Finetuning script for 2D Llama-Unet model.
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
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import einops
import sys
from train.train_tokenizer_2d import TokenizerModel
from copy import deepcopy
import wandb
from functools import reduce
from peft import get_peft_model, LoraConfig, TaskType
import random
from utils.load_data import get_data
from utils.load_data_new import load_wave, load_vort, load_rd

class ZebraModel(pl.LightningModule):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.save_hyperparameters()

        config = LlamaConfig(**cfg.model)
        self.cfg = cfg
        self.model = LlamaForCausalLM(config)
        self.tokenizer = tokenizer #.cuda()
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
        self.accumulate_grad = 1
        self.custom_step = 0
        self.noise_level = 1e-4
        self.use_peft = True

    def forward(self, x):
        return self.model(x)
    

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
                                    ch_mults= (1, 1, 2, 2), #(1, 2, 2, 2),
                                    is_attn=(False, False, False, False),
                                    mid_attn = False, #False
                                    n_blocks= 2, #n_blocks=1
                                    param_conditioning = None,
                                    use_scale_shift_norm=False, #False
                                    use1x1 = False,
                                    code_dim=self.cfg.model.hidden_size).to(self.model.device)
        if self.use_peft:
            peft_config = LoraConfig(
                lora_alpha = 16,
                lora_dropout = 0.1,
                r  = 128,
                bias = "none",
                task_type = "FEATURE_EXTRACTION")

            self.model = get_peft_model(self.model, peft_config)

    def training_step(self, batch, batch_idx):

        images, sequences, context_sequences = batch
        sequences = rearrange(sequences, 'b c h w t -> b (t h w c)')
        context_sequences = rearrange(context_sequences, 'b k c h w t -> b k t (h w c)')

        opt = self.optimizers()
        sch = self.lr_schedulers()
        t = images.shape[-1]
        b = sequences.shape[0]
        k = context_sequences.shape[1]
        device = sequences.device

        num_examples = random.randint(1, k)
        cat_context_seq = []
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

            cat_context_seq.append(context_seq)

        padded_chunks = [torch.tensor(chunk) for chunk in cat_context_seq]
        input_ids = torch.stack(padded_chunks).to(device)

        if self.use_peft:
            embed = self.model.model.model.embed_tokens(input_ids)
        else:
            embed = self.model.model.embed_tokens(input_ids)

        embed = torch.cat([embed, self.context_token[None, ...].repeat(b,1,1)], axis=1)
        out = self.model(inputs_embeds=embed, output_hidden_states=True)
        emb = out.hidden_states[-1][:, -self.num_codes:, :]  # (b h) we take the hidden

        emb = einops.repeat(emb, 'b h c -> (b t) h c', t=t)
        #emb = rearrange(emb, 'b t h -> (b t) h')
        x = rearrange(images, 'b c h w t -> (b t) c h w')
        noise = torch.randn_like(x)*self.noise_level
        pred = self.neural_operator(x+noise, z=emb)
        pred = rearrange(pred, '(b t) c h w -> b c h w t', t=t)
        loss = self.rel_loss(pred[..., :-1], images[..., 1:]).mean()
      
        #out = self.model(input_ids, labels=labels)
        #loss = out.loss
        
        #opt.zero_grad()
        self.manual_backward(loss/self.accumulate_grad)

        if (self.custom_step + 1) % self.accumulate_grad == 0:
            opt.step()
            opt.zero_grad() 
            sch.step()
        
        self.custom_step+=1
    
        self.log('train_relative_l2_step', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        images, sequences, context_sequences = batch
        sequences = rearrange(sequences, 'b c h w t -> b (t h w c)')
        context_sequences = rearrange(context_sequences, 'b k c h w t -> b k t (h w c)')

        opt = self.optimizers()
        sch = self.lr_schedulers()
        t = images.shape[-1]
        b = sequences.shape[0]
        k = context_sequences.shape[1]
        device = sequences.device

        num_examples = random.randint(1, k)
        cat_context_seq = []
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

            cat_context_seq.append(context_seq)

        padded_chunks = [torch.tensor(chunk) for chunk in cat_context_seq]
        input_ids = torch.stack(padded_chunks).to(device)
        
        if self.use_peft:
            embed = self.model.model.model.embed_tokens(input_ids)
        else:
            embed = self.model.model.embed_tokens(input_ids)
        embed = torch.cat([embed, self.context_token[None, ...].repeat(b,1,1)], axis=1)
        out = self.model(inputs_embeds=embed, output_hidden_states=True)
        emb = out.hidden_states[-1][:, -self.num_codes:, :]  # (b h) we take the hidden

        #out = self.model(input_ids, labels=labels)
        #loss = out.loss
        #print('emb', emb.shape)

        x = images[..., 0]
        u_pred = []
        for time in range(t-1):
            pred = self.neural_operator(x, z=emb)
            x = pred.detach()
            u_pred.append(x)
            #print('x', time, x.shape)

        pred = torch.stack(u_pred, axis=-1)
        #print('pred', pred.shape, 'image', image.shape)
        loss = self.rel_loss(pred, images[..., 1:]).mean()

        self.log('val_relatve_l2', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_nll', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        #embedding_params = self.model.get_input_embeddings().weight
        #new_embedding_params = embedding_params[-256:]
        if not self.use_peft:
            self.model.model.embed_tokens.requires_grad_(False)
        lr = self.cfg.train.learning_rate
        optimizer = AdamW([{"params": self.model.parameters()},
                            {"params": self.neural_operator.parameters()},
                            {"params": self.context_token}],
                            lr=lr, weight_decay=1e-4)
        #optimizer = AdamW([{"params": self.model.parameters()}], lr=self.cfg.train.learning_rate, weight_decay=1e-3) #1e-6
        #optimizer = AdamW(self.model.parameters(), lr=self.cfg.train.learning_rate, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.train.max_steps//self.accumulate_grad)
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

  
class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, u, slice_size=20, num_context_images=5):
        self.dataset = dataset
        self.u = u 
        self.slice_size = slice_size
        self.num_context_images = num_context_images
        self.subsampling_t = 10
        self.random_t = False

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        sub_t = 3
    
        images = self.u[idx, ..., ::sub_t] 

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

        return self.dataset.__getitem__(idx)[..., ::sub_t], images, context_images

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

    batch_size=32
    if dataset_name=="wave2d":
        train_loader, val_loader, test_loader = load_wave("/lustre/fsn1/projects/rech/mdw/ueg82cz/", batch_size, batch_size, slice_size=30, shuffle=False)
    elif dataset_name=="vorticity":
        train_loader, val_loader, test_loader = load_vort("/lustre/fsn1/projects/rech/mdw/ueg82cz/", batch_size, batch_size, slice_size=30, shuffle=False) 
    elif dataset_name=="rd":
        train_loader, val_loader, test_loader = load_rd("/lustre/fsn1/projects/rech/mdw/ueg82cz/", batch_size, batch_size, slice_size=20, shuffle=False) 
    
    # retrieve the trainset 
 
    u_trainset, u_valset, u_testset = train_loader.dataset, val_loader.dataset, test_loader.dataset

    #vae_ckpt_path = f"{cfg.dataset.input_dir}/{dataset_name}/{run_name}.ckpt"

    if dataset_name == "vorticity":
        zebra_ckpt_path = "/lustre/fsn1/projects/rech/mdw/ueg82cz/zebra/transformer/vorticity/step37000.ckpt"
    elif dataset_name == "wave2d":
        zebra_ckpt_path = "/lustre/fsn1/projects/rech/mdw/ueg82cz/zebra/transformer/wave2d/last-v9.ckpt"
    else:
        raise ValueError("Dataset not supported")

    model = ZebraModel.load_from_checkpoint(zebra_ckpt_path).cuda()
    model.set_no_decoder()
    tokenizer = model.tokenizer.eval()

    #tkn = TokenizerModel.load_from_checkpoint(vae_ckpt_path)
    #tokenizer = tkn.model.float().eval()

    token_dataset_path = f"/lustre/fsn1/projects/rech/mdw/ueg82cz/{dataset_name}"

    try:
        token_train = torch.load(f"{token_dataset_path}/train_token_{run_name}.pt")
        token_val = torch.load(f"{token_dataset_path}/val_token_{run_name}.pt")
        token_test = torch.load(f"{token_dataset_path}/test_token_{run_name}.pt")
    
    except: 
     
        token_train = []
        token_val = []
        token_test = []
        
        with torch.no_grad():
            for out_list, loader in zip([token_train, token_val, token_test], [train_loader, val_loader, test_loader]):
                for batch in loader:
                    sequences = batch.cuda()
                    t = sequences.shape[-1]
                    sequences = rearrange(sequences, "b c h w t-> (b t) c h w")
                    codes, indices = tokenizer.tokenizer(sequences, return_codes=True)
                    indices = rearrange(indices, "(b t) c h w -> b c h w t", t=t)
                    out_list.append(indices.cpu().detach()) 

        token_train = torch.cat(token_train, axis=0)
        token_val = torch.cat(token_val, axis=0)
        token_test = torch.cat(token_test, axis=0)
        
        os.makedirs(token_dataset_path, exist_ok=True)
        torch.save(token_train, f"{token_dataset_path}/train_token_{run_name}.pt")
        torch.save(token_val, f"{token_dataset_path}/val_token_{run_name}.pt")
        torch.save(token_test, f"{token_dataset_path}/test_token_{run_name}.pt")
    
    #if cfg.dataset.resume_training:
        #zebra_ckpt_path = cfg.dataset.checkpoint_path # WARNING to do in config 
        #model = ZebraModel.load_from_checkpoint(zebra_ckpt_path)
        
    #else:
    #    model = ZebraModel(tokenizer, cfg).cuda()

    slice_size=cfg.dataset.slice_size
    num_context_images = cfg.dataset.num_context_images
    trainset = TemporalDataset(u_trainset, token_train, slice_size=slice_size, num_context_images=num_context_images)
    testset = TemporalDataset(u_testset, token_test, slice_size=slice_size, num_context_images=num_context_images)

    run = wandb.init(project="zebra")
    run_name = wandb.run.name
    run.tags = (
            ("llama_unet",)
            + (dataset_name,)
        )
    zebra_ckpt_dir = f"{cfg.dataset.output_dir}/{dataset_name}/{run_name}/"
    batch_size = 8 # so that if each sample has only 0 example, it still makes 12*128*10 samples for wave 2d

    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,  prefetch_factor=1)# collate_fn=tokenize_function)
    val_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=1)#, collate_fn=tokenize_function)
    checkpoint_callback = ModelCheckpoint(dirpath=zebra_ckpt_dir, save_top_k=2, save_last=True, verbose=True, every_n_train_steps=1000, filename='{step}-{train_relative_l2_step:.4f}', monitor="train_relative_l2_step")
    wandb_logger = WandbLogger(project="zebra", tags=[cfg.dataset.dataset_name, cfg.dataset.trainset, "token"])
    trainer = pl.Trainer(max_steps=cfg.train.max_steps, devices="auto", accumulate_grad_batches=1, accelerator="gpu", logger=wandb_logger, callbacks=[checkpoint_callback], check_val_every_n_epoch=10, precision="16-mixed") # 16-mixed
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
