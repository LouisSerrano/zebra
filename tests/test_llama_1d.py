import os
import hydra
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, LlamaConfig
import random
import sys

# Add project path
sys.path.append('/home/serrano/Projects/zebra/')
from train.train_tokenizer_vqgan_mppde import TokenizerModel
from utils.load_data import get_data
from test_utils import get_test_metrics_1d, ResultsManager, create_comparison_plot

class ZebraModel(pl.LightningModule):
    """Simplified Zebra model class for 1D testing."""
    def __init__(self, tokenizer, cfg):
        super().__init__()
        config = LlamaConfig(**cfg.model)
        self.cfg = cfg
        self.model = LlamaForCausalLM(config).float()
        self.tokenizer = tokenizer.cuda().float()
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

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW([{"params": self.model.parameters()}], lr=self.cfg.train.learning_rate, weight_decay=1e-6)
        scheduler = ConstantWithLinearDecayScheduler(optimizer, int(0.8*self.cfg.train.max_steps), self.cfg.train.max_steps)
        return [optimizer], [scheduler]


class ConstantWithLinearDecayScheduler(_LRScheduler):
    """Learning rate scheduler with constant then linear decay."""
    def __init__(self, optimizer, constant_steps, total_steps, last_epoch=-1):
        self.constant_steps = constant_steps
        self.total_steps = total_steps
        super(ConstantWithLinearDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.constant_steps:
            return self.base_lrs
        else:
            decay_steps = self.total_steps - self.constant_steps
            decay_factor = (self.total_steps - self.last_epoch) / decay_steps
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class TemporalDataset(torch.utils.data.Dataset):
    """Simplified temporal dataset for 1D sequences."""
    def __init__(self, u, slice_size=20, subsampling_t=10, num_context_images=5, context_slice_size=10):
        self.u = u 
        self.slice_size = slice_size
        self.context_slice_size = context_slice_size
        self.num_context_images = num_context_images
        self.subsampling_t = subsampling_t

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        images = self.u[idx, ..., ::self.subsampling_t]

        max_start_index = images.shape[-1] - self.slice_size
        if max_start_index < 0:
            raise ValueError("Slice size is larger than the sequence length.")
        start_index = 0
        images = images[..., start_index:start_index + self.slice_size]

        min_index = (idx // 10) * 10  # 10 trajectories per environment
        max_index = min_index + 9

        context_images = []
        ctx_idx_list = [i for i in range(min_index, max_index + 1)] 
        ctx_idx_list.remove(idx)
        random.shuffle(ctx_idx_list)
        for j in range(self.num_context_images):
            ctx_idx = ctx_idx_list[j]
            context_images.append(self.u[ctx_idx, ..., ::self.subsampling_t][..., start_index:start_index + self.context_slice_size])

        context_images = torch.stack(context_images)
        return images.float(), context_images.float()

@hydra.main(config_path="../config", config_name="llama_burgers.yaml")
def main(cfg):
    """Simplified 1D Llama testing main function."""
    
    # Dataset configuration
    dataset_name = cfg.dataset.dataset_name
    run_name_dict = {
        "combined_equation": "misunderstood-armadillo-2084",
        "wave_varying_boundary": "lively-capybara-1714",
        "wave_fixed_boundary": "gentle-dragon-1701",
        "heat2": "genial-bush-1685",
        "advection": "gentle-firebrand-1973",
        "burgers_nu_forcing2": "spring-planet-1689"
    }

    run_name = run_name_dict[dataset_name]
    u_train, u_val, u_test = get_data(dataset_name)
    
    # Configure dataset parameters
    include_special_token = False
    num_token = 2
    
    if dataset_name == "wave_varying_boundary":
        sub_t, num_context, slice_size = 4, 3, 15
    elif dataset_name == "wave_fixed_boundary":
        sub_t, num_context, slice_size = 1, 3, 10
    else:
        sub_t, num_context, slice_size = 10, 5, 10

    if dataset_name == 'combined_equation':
        include_special_token = True
        num_token = 2

    # Create datasets and data loader
    testset = TemporalDataset(
        torch.from_numpy(u_test).float(), 
        slice_size=slice_size, 
        subsampling_t=sub_t, 
        num_context_images=num_context, 
        context_slice_size=slice_size
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=testset, 
        batch_size=10, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True, 
        prefetch_factor=1
    )

    # Load model
    zebra_ckpt_path = f"/data/serrano/zebra_results/zebra_patch/{dataset_name}/{run_name}/last.ckpt"
    model = ZebraModel.load_from_checkpoint(zebra_ckpt_path).cuda().eval()
    
    # Initialize results manager
    results_manager = ResultsManager()
    
    # Run test with different configurations
    test_configs = [
        {"inp_size": 1, "num_examples": 0, "name": "zeroshot"},
        {"inp_size": 1, "num_examples": 1, "name": "oneshot"},
        {"inp_size": 1, "num_examples": 5, "name": "fiveshot"}
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\nRunning {config['name']} test...")
        
        l2_loss, predictions, ground_truth, context = get_test_metrics_1d(
            val_loader, 
            model, 
            num_token=num_token, 
            inp_size=config["inp_size"], 
            num_examples=config["num_examples"], 
            include_special_token=include_special_token
        )
        
        result = {
            "dataset": dataset_name,
            "experiment": config["name"],
            "inp_size": config["inp_size"],
            "num_examples": config["num_examples"],
            "l2_loss": l2_loss
        }
        
        all_results.append(result)
        print(f"L2 Loss: {l2_loss:.6f}")
        
        # Save results
        results_manager.save_predictions(
            predictions, ground_truth, context, 
            dataset_name, config["name"]
        )
        
        # Create and save plot
        fig = create_comparison_plot(
            ground_truth, predictions, context, 
            title=f"{dataset_name} - {config['name']} Results"
        )
        results_manager.save_plot(fig, dataset_name, config["name"], "comparison")
    
    # Save metrics summary
    metrics_file = results_manager.save_metrics(all_results, dataset_name, "1d_test_summary")
    print(f"\nResults saved to: {metrics_file}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for result in all_results:
        print(f"{result['experiment']}: L2 Loss = {result['l2_loss']:.6f}")

if __name__ == "__main__":
    main()
