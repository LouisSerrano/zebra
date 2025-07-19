"""
Shared utilities for zebra model testing.
This module contains common functions used across different test scenarios.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from einops import rearrange
from pathlib import Path
import properscoring as ps
import uncertainty_toolbox as uct
from datetime import datetime


class RelativeL2(nn.Module):
    """Relative L2 loss function."""
    def forward(self, x, y):
        x = rearrange(x, "b ... -> b (...)")
        y = rearrange(y, "b ... -> b (...)")
        diff_norms = torch.linalg.norm(x - y, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        return (diff_norms / y_norms).mean()


class ResultsManager:
    """Manages saving and organizing test results."""
    
    def __init__(self, base_dir="/mnt/home/lserrano/zebra/results"):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_save_path(self, result_type, dataset_name, experiment_name=None):
        """Get standardized save path for different result types."""
        path = self.base_dir / result_type / dataset_name
        if experiment_name:
            path = path / experiment_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_metrics(self, metrics_dict, dataset_name, experiment_name):
        """Save metrics to CSV with timestamp."""
        path = self.get_save_path("metrics", dataset_name, experiment_name)
        filename = f"metrics_{self.timestamp}.csv"
        
        df = pd.DataFrame(metrics_dict)
        df.to_csv(path / filename, index=False)
        return path / filename
    
    def save_predictions(self, predictions, ground_truth, context, dataset_name, experiment_name):
        """Save model predictions and ground truth."""
        path = self.get_save_path("predictions", dataset_name, experiment_name)
        
        np.save(path / f"predictions_{self.timestamp}.npy", predictions)
        np.save(path / f"ground_truth_{self.timestamp}.npy", ground_truth)
        np.save(path / f"context_{self.timestamp}.npy", context)
        
        return path
    
    def save_plot(self, fig, dataset_name, experiment_name, plot_name):
        """Save matplotlib figure."""
        path = self.get_save_path("plots", dataset_name, experiment_name)
        filename = f"{plot_name}_{self.timestamp}.pdf"
        fig.savefig(path / filename, bbox_inches='tight')
        plt.close(fig)
        return path / filename


def get_test_metrics_1d(loader, model, num_token=2, inp_size=1, num_examples=0, include_special_token=False):
    """Simplified 1D test metrics computation."""
    rel_loss = RelativeL2()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_ground_truth = []
    all_context = []

    with torch.no_grad():
        for images, context_images in loader:
            images = images.cuda()
            context_images = context_images.cuda()

            t = images.shape[-1]
            t_context = context_images.shape[-1]
            b = images.shape[0]
            device = images.device

            # Tokenize images
            x = rearrange(images, "b h c t -> (b t) c h").float()
            codes, indices = model.tokenizer.tokenizer(x, return_codes=True)
            h_lat = indices.shape[1]
            sequences = rearrange(indices, '(b t) h c -> b (t h c)', t=t)

            # Tokenize context
            k = context_images.shape[1]
            x_context = rearrange(context_images, "b k h c t -> (b k t) c h").float()
            codes, indices = model.tokenizer.tokenizer(x_context, return_codes=True)
            context_sequences = rearrange(indices, '(b k t) h c -> b k (t h c)', t=t_context, k=k)

            # Build input sequence
            inp = torch.cat([torch.full((b,1), model.bot_token_id, device=device), 
                           sequences[..., :num_token*inp_size*16]], axis=1)
            
            if num_examples > 0:
                context = torch.cat([torch.full((b, k, 1), model.bot_token_id, device=device), 
                                   context_sequences, 
                                   torch.full((b, k, 1), model.eot_token_id, device=device)], axis=2)
                context = rearrange(context[:, :num_examples], 'b k t-> b (k t)')
                input_ids = torch.cat([torch.full((b,1), model.bos_token_id, device=device),
                                     context, inp], axis=1)
            else:
                input_ids = torch.cat([torch.full((b,1), model.bos_token_id, device=device), inp], axis=1)

            # Generate predictions
            rollout_len = t - inp_size
            output_len = rollout_len*h_lat*num_token + 1
            max_length = input_ids.shape[1] + output_len
            
            output = model.model.generate(
                input_ids, 
                max_length=max_length, 
                num_return_sequences=1, 
                do_sample=True, 
                temperature=0.1,
                bad_words_ids=[[256],[257],[258],[259],[260],[261],[262]]
            )
            
            # Decode predictions
            indices = rearrange(output[:, -output_len:-1], 'b (t h c) -> (b h t) c', 
                              t=rollout_len, h=h_lat).cuda()
            indices = torch.clamp(indices, 0, 255)
            quantized_pred = model.tokenizer.tokenizer.quantizers.get_output_from_indices(indices)
            quantized_pred = rearrange(quantized_pred, '(b h t) c -> (b t) c h', t=rollout_len, h=h_lat)
            upred = model.tokenizer.tokenizer.decode(quantized_pred, cond=None)
            upred = rearrange(upred, '(b t) c h -> b h c t', t=rollout_len)

            # Compute loss
            loss = rel_loss(upred, images[..., inp_size:])
            total_loss += loss * b
            total_samples += b
            
            # Store results
            all_predictions.append(upred.cpu().detach().numpy())
            all_ground_truth.append(images.cpu().detach().numpy())
            all_context.append(context_images.cpu().detach().numpy())

    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_predictions)
    ground_truth = np.concatenate(all_ground_truth)
    context = np.concatenate(all_context)
    
    return avg_loss.item(), predictions, ground_truth, context


def get_test_metrics_2d(loader, model, input_size=1, num_examples=0):
    """Simplified 2D test metrics computation."""
    rel_loss = RelativeL2()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_ground_truth = []
    all_context = []

    with torch.no_grad():
        for images, context_images, tokens, context_tokens in loader:
            images = images.cuda()
            
            sequences, context_sequences = tokens, context_tokens 
            sequences = rearrange(sequences[..., :input_size], 'b c h w t -> b (t h w c)')
            context_sequences = rearrange(context_sequences, 'b k c h w t -> b k t (h w c)')

            t = sequences.shape[-1]
            b = sequences.shape[0]
            k = context_sequences.shape[1]
            device = sequences.device

            # Build input sequence
            cat_context_seq = []
            for j in range(b):
                context_seq = [model.bos_token_id]
                trajectory_size = t
                for ex in range(num_examples):
                    context_seq.append(model.bot_token_id)
                    context_seq.extend(rearrange(context_sequences[j, ex, :trajectory_size], 't h -> (t h)').tolist())
                    context_seq.append(model.eot_token_id)

                context_seq.append(model.bot_token_id)
                context_seq.extend(sequences[j].tolist())
                cat_context_seq.append(context_seq)
            
            input_ids = torch.tensor(cat_context_seq).cuda()
            
            # Generate predictions
            rollout_len = 9
            output_len = 256*rollout_len
            max_length = output_len + input_ids.shape[1]
            
            output = model.model.generate(
                input_ids, 
                max_length=max_length, 
                num_return_sequences=1, 
                do_sample=True, 
                bad_words_ids=[[2048],[2049],[2050],[2051],[2052],[2053],[2054]]
            )
            
            # Decode predictions
            indices = rearrange(output[:, -output_len:], 'b (t h w c)  -> (b t h w) c' , 
                              t=rollout_len, h=16, w=16).cuda()
            quantized_pred = model.tokenizer.tokenizer.quantizers.get_output_from_indices(indices)
            quantized_pred = rearrange(quantized_pred, '(b t h w) c -> (b t) c h w', t=rollout_len, h=16, w=16)
            
            upred = model.tokenizer.tokenizer.decode(quantized_pred, cond=None)
            upred = rearrange(upred, '(b t) c h w -> b c h w t', t=rollout_len)
            
            # Compute loss
            loss = rel_loss(upred, images[..., 1:])
            total_loss += loss * b
            total_samples += b
            
            # Store results
            all_predictions.append(upred.cpu().detach().numpy())
            all_ground_truth.append(images.cpu().detach().numpy())
            all_context.append(context_images[:, :num_examples].cpu().detach().numpy())

    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_predictions)
    ground_truth = np.concatenate(all_ground_truth)
    context = np.concatenate(all_context)
    
    return avg_loss.item(), predictions, ground_truth, context


def get_uncertainty_metrics(loader, model, num_token=2, temperature=1.0, inp_size=1, 
                          num_examples=0, include_special_token=False, n_samples=20):
    """Compute uncertainty metrics with multiple samples."""
    rel_loss = RelativeL2()
    total_loss = 0
    total_samples = 0
    total_crps = 0
    total_rmsce = 0
    total_accuracy = 0
    total_rel_std = 0

    all_means = []
    all_stds = []
    all_ground_truth = []
    all_context = []

    with torch.no_grad():
        for images, context_images in loader:
            images = images.cuda()
            context_images = context_images.cuda()

            t = images.shape[-1]
            t_context = context_images.shape[-1]
            b = images.shape[0]
            device = images.device

            # Tokenize (same as 1D case)
            x = rearrange(images, "b h c t -> (b t) c h").float()
            codes, indices = model.tokenizer.tokenizer(x, return_codes=True)
            h_lat = indices.shape[1]
            sequences = rearrange(indices, '(b t) h c -> b (t h c)', t=t)

            k = context_images.shape[1]
            x_context = rearrange(context_images, "b k h c t -> (b k t) c h").float()
            codes, indices = model.tokenizer.tokenizer(x_context, return_codes=True)
            context_sequences = rearrange(indices, '(b k t) h c -> b k (t h c)', t=t_context, k=k)

            # Build input
            inp = torch.cat([torch.full((b,1), model.bot_token_id, device=device), 
                           sequences[..., :num_token*inp_size*16]], axis=1)
            
            if num_examples > 0:
                context = torch.cat([torch.full((b, k, 1), model.bot_token_id, device=device), 
                                   context_sequences, 
                                   torch.full((b, k, 1), model.eot_token_id, device=device)], axis=2)
                context = rearrange(context[:, :num_examples], 'b k t-> b (k t)')
                input_ids = torch.cat([torch.full((b,1), model.bos_token_id, device=device),
                                     context, inp], axis=1)
            else:
                input_ids = torch.cat([torch.full((b,1), model.bos_token_id, device=device), inp], axis=1)

            # Generate multiple samples
            rollout_len = t - inp_size
            output_len = rollout_len*h_lat*num_token + 1
            max_length = input_ids.shape[1] + output_len
            
            output = model.model.generate(
                input_ids, 
                max_length=max_length, 
                temperature=temperature, 
                num_return_sequences=n_samples,
                do_sample=True, 
                bad_words_ids=[[256],[257],[258],[259],[260],[261],[262]]
            )
            
            output = rearrange(output, '(b c) t -> c b t', c=n_samples)
            
            # Decode all samples
            upred_list = []
            for sample_idx in range(n_samples):
                indices = rearrange(output[sample_idx][:, -output_len:-1], 'b (t h c) -> (b h t) c', 
                                  t=rollout_len, h=h_lat).cuda()
                indices = torch.clamp(indices, 0, 255)
                quantized_pred = model.tokenizer.tokenizer.quantizers.get_output_from_indices(indices)
                quantized_pred = rearrange(quantized_pred, '(b h t) c -> (b t) c h', t=rollout_len, h=h_lat)
                upred = model.tokenizer.tokenizer.decode(quantized_pred, cond=None)
                upred = rearrange(upred, '(b t) c h -> b h c t', t=rollout_len)
                upred_list.append(upred)

            predictions = torch.stack(upred_list)  # Shape: (n_samples, b, h, c, t)

            # Compute statistics
            mean = predictions.mean(0)
            std = predictions.std(0) + 1e-6

            # Compute metrics
            crps_batch = ps.crps_ensemble(images[..., 1:].cpu().reshape(-1), 
                                        rearrange(predictions, "n ... -> ... n").reshape(-1, n_samples).cpu())

            rmsce_batch = uct.metrics_calibration.root_mean_squared_calibration_error(
                mean.reshape(-1).cpu().numpy(),
                std.reshape(-1).cpu().numpy(),
                images[..., 1:].reshape(-1).cpu().numpy()
            )

            # Coverage accuracy (3-sigma bounds)
            lower_bound = mean[..., -1] - 3*std[..., -1]
            upper_bound = mean[..., -1] + 3*std[..., -1]
            outside_bounds = torch.logical_or(images[..., -1].cpu() < lower_bound.cpu(), 
                                            images[..., -1].cpu() > upper_bound.cpu()).float()
            accuracy = (1 - outside_bounds).mean()

            # Relative standard deviation
            sigma = rearrange(std, "b ... -> b (...)")
            y = rearrange(mean, "b ... -> b (...)")
            sigma_norms = torch.linalg.norm(sigma, ord=2, dim=-1)
            y_norms = torch.linalg.norm(y, ord=2, dim=-1)
            rel_std = (sigma_norms / y_norms).mean()

            loss = rel_loss(mean, images[..., inp_size:])
            
            # Accumulate metrics
            total_loss += loss * b
            total_crps += crps_batch.mean() * b
            total_rmsce += rmsce_batch * b
            total_accuracy += accuracy * b
            total_rel_std += rel_std * b
            total_samples += b
            
            # Store results
            all_means.append(mean.cpu().detach().numpy())
            all_stds.append(std.cpu().detach().numpy())
            all_ground_truth.append(images.cpu().detach().numpy())
            all_context.append(context_images.cpu().detach().numpy())

    # Compute averages
    metrics = {
        'l2_loss': (total_loss / total_samples).item(),
        'rel_std': (total_rel_std / total_samples).item(),
        'accuracy': (total_accuracy / total_samples).item(),
        'crps': (total_crps / total_samples).item(),
        'rmsce': (total_rmsce / total_samples).item()
    }
    
    results = {
        'means': np.concatenate(all_means),
        'stds': np.concatenate(all_stds),
        'ground_truth': np.concatenate(all_ground_truth),
        'context': np.concatenate(all_context)
    }
    
    return metrics, results


def create_comparison_plot(ground_truth, predictions, context=None, save_path=None, 
                         title="Model Predictions", idx=0):
    """Create a simple comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    if ground_truth.ndim == 4:  # 1D case: (b, h, c, t)
        for t in range(min(ground_truth.shape[-1], 10)):
            axes[0].plot(ground_truth[idx, :, 0, t], label=f'Time {t}')
    axes[0].set_title('Ground Truth')
    axes[0].legend()
    axes[0].grid(True)
    
    # Predictions
    if predictions.ndim == 4:  # 1D case
        for t in range(min(predictions.shape[-1], 10)):
            axes[1].plot(predictions[idx, :, 0, t], label=f'Time {t}')
    axes[1].set_title('Predictions')
    axes[1].legend()
    axes[1].grid(True)
    
    # Overlay
    if ground_truth.ndim == 4:
        t_final = min(ground_truth.shape[-1], predictions.shape[-1]) - 1
        axes[2].plot(ground_truth[idx, :, 0, t_final], 'r-', label='Ground Truth', linewidth=2)
        axes[2].plot(predictions[idx, :, 0, t_final], 'b--', label='Prediction', linewidth=2)
    axes[2].set_title('Final Time Comparison')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    return fig