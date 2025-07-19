import os
import hydra
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, LlamaConfig
import random
import sys
from time import time

# Add project path
WORK_DIR = os.environ.get('WORK', '/home/serrano/Projects')
sys.path.append(f"{WORK_DIR}/zebra/")
from train.train_tokenizer_2d import TokenizerModel
from utils.load_data_new import load_wave, load_vort
from test_utils import get_test_metrics_2d, ResultsManager, create_comparison_plot 

class ZebraModel(pl.LightningModule):
    """Simplified Zebra model class for 2D testing."""
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.save_hyperparameters()
        
        config = LlamaConfig(**cfg.model)
        self.cfg = cfg
        self.model = LlamaForCausalLM(config)
        self.tokenizer = tokenizer
        self.automatic_optimization = False
 
        self.bos_token_id = cfg.model.bos_token_id
        self.eos_token_id = cfg.model.eos_token_id
        self.context_token_id = cfg.model.context_token_id
        self.input_token_id = cfg.model.input_token_id
        self.target_token_id = cfg.model.target_token_id
        self.bot_token_id = cfg.model.bot_token_id
        self.eot_token_id = cfg.model.eot_token_id
        self.pad_token_id = cfg.model.pad_token_id
        self.target_size = self.cfg.dataset.slice_size
        self.sample_context = False
        self.accumulate_grad = 4
        self.custom_step = 0

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW([{"params": self.model.parameters()}], lr=self.cfg.train.learning_rate, weight_decay=1e-6)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=self.cfg.train.max_steps//4)
        return [optimizer], [scheduler]


class TemporalDataset(torch.utils.data.Dataset):
    """Simplified temporal dataset for 2D sequences."""
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
        sub_t = 3
        images = self.u[idx, ..., ::sub_t]
        tokens = self.token[idx, ..., ::sub_t]

        start_index = 0 
        min_index = (idx // 10) * 10  # 10 trajectories per environment
        max_index = min_index + 9

        context_images = []
        context_tokens = []
        max_start_index = images.shape[-1] - self.slice_size
        ctx_idx_list = [i for i in range(min_index, max_index + 1)] 
        ctx_idx_list.remove(idx)
        random.shuffle(ctx_idx_list)
        
        for j in range(self.num_context_images):
            if self.random_t:
                start_index = np.random.randint(0, max_start_index + 1)
            ctx_idx = ctx_idx_list[j]
            context_images.append(self.u[ctx_idx, ..., ::sub_t][..., start_index:start_index + self.slice_size])
            context_tokens.append(self.token[ctx_idx, ..., ::sub_t][..., start_index:start_index + self.slice_size])

        context_images = torch.stack(context_images)
        context_tokens = torch.stack(context_tokens)
        return images, context_images, tokens, context_tokens

@hydra.main(config_path="../config", config_name="llama_2d.yaml")
def main(cfg):
    """Simplified 2D Llama testing main function."""
    
    dataset_name = cfg.dataset.dataset_name
    run_name = cfg.dataset.run_name

    # Load data
    batch_size = 32
    if dataset_name == "wave2d":
        train_loader, val_loader, test_loader = load_wave(
            "/lustre/fsn1/projects/rech/mdw/ueg82cz/", 
            batch_size, batch_size, slice_size=30, shuffle=False
        )
    elif dataset_name == "vorticity":
        train_loader, val_loader, test_loader = load_vort(
            "/lustre/fsn1/projects/rech/mdw/ueg82cz/", 
            batch_size, batch_size, slice_size=30, shuffle=False
        )

    # Load tokenizer
    vae_ckpt_path = f"{cfg.dataset.input_dir}/{dataset_name}/{run_name}.ckpt"
    tkn = TokenizerModel.load_from_checkpoint(vae_ckpt_path)
    tokenizer = tkn.model.float().eval()
    
    # Tokenize validation data
    token_val = []
    u_val = []
    
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch.cuda()
            u_val.append(sequences.cpu())
            t = sequences.shape[-1]
            sequences = rearrange(sequences, "b c h w t-> (b t) c h w")
            codes, indices = tokenizer.tokenizer(sequences, return_codes=True)
            indices = rearrange(indices, "(b t) c h w -> b c h w t", t=t)
            token_val.append(indices.cpu().detach()) 

    token_val = torch.cat(token_val, axis=0)
    u_val = torch.cat(u_val, axis=0) 
        
    # Load model
    zebra_ckpt_path = cfg.dataset.checkpoint_path
    model = ZebraModel.load_from_checkpoint(zebra_ckpt_path)
        
    # Create dataset
    slice_size = cfg.dataset.slice_size
    num_context_images = cfg.dataset.num_context_images
    testset = TemporalDataset(u_val, token_val, slice_size=slice_size, num_context_images=num_context_images)

    val_loader = torch.utils.data.DataLoader(
        dataset=testset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True, 
        prefetch_factor=1
    )
    
    # Initialize results manager
    results_manager = ResultsManager()
    
    # Run test with different configurations
    test_configs = [
        {"input_size": 1, "num_examples": 0, "name": "zeroshot"},
        {"input_size": 1, "num_examples": 1, "name": "oneshot"},
        {"input_size": 1, "num_examples": 3, "name": "threeshot"}
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\nRunning {config['name']} test...")
        start_time = time()
        
        l2_loss, predictions, ground_truth, context = get_test_metrics_2d(
            val_loader, 
            model, 
            input_size=config["input_size"], 
            num_examples=config["num_examples"]
        )
        
        result = {
            "dataset": dataset_name,
            "experiment": config["name"],
            "input_size": config["input_size"],
            "num_examples": config["num_examples"],
            "l2_loss": l2_loss,
            "time": time() - start_time
        }
        
        all_results.append(result)
        print(f"L2 Loss: {l2_loss:.6f}, Time: {result['time']:.2f}s")
        
        # Save results
        results_manager.save_predictions(
            predictions, ground_truth, context, 
            dataset_name, config["name"]
        )
    
    # Save metrics summary
    metrics_file = results_manager.save_metrics(all_results, dataset_name, "2d_test_summary")
    print(f"\nResults saved to: {metrics_file}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for result in all_results:
        print(f"{result['experiment']}: L2 Loss = {result['l2_loss']:.6f}, Time = {result['time']:.2f}s")

if __name__ == "__main__":
    main()
