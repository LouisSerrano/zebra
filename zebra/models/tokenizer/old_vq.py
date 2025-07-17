import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.1):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

class VectorQuantizerClassic(nn.Module):
    def __init__(self, n_e=2048,
                        e_dim=8,
                        beta=0.25,
                        entropy_loss_ratio=0.1,
                        l2_norm=False,
                        show_usage=True,
                        codebook_usage_weight=1.0,
                        ema_update=False,
                        ema_decay=0.99):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.codebook_usage_weight = codebook_usage_weight
        self.ema_update = ema_update
        self.ema_decay = ema_decay

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(n_e)))
    
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        if z.ndim == 3:
            z = torch.einsum('b c h -> b h c', z).contiguous()

        elif z.ndim==4:
            z = torch.einsum('b c h w -> b h w c', z).contiguous()

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        print('d', d.shape)

        min_encoding_indices = torch.argmin(d, dim=1)
        #print("min_encoding_indices", min_encoding_indices.shape)
        #print("embedding", embedding.shape)
        z_q = torch.gather(embedding, 0, min_encoding_indices.unsqueeze(1).expand(-1, embedding.shape[1])).view(z.shape)
        #z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = 0 #None
        min_encodings = 0 #None
        vq_loss = 0 #None
        commit_loss = 0 #None
        entropy_loss = 0 #None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            #self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            #self.codebook_used[-cur_len:] = min_encoding_indices
            self.codebook_used = torch.unique(torch.cat([torch.unique(min_encoding_indices), torch.unique(self.codebook_used)]))
            codebook_usage = len(self.codebook_used) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

            if self.ema_update:
                self.codebook_avg.data = self.ema_decay * self.embedding.weight.data + (1 - self.ema_decay) * self.embedding.weight.data
                self.embedding.weight.data = self.codebook_avg.data
                commit_loss = self.beta * torch.mean((self.embedding.weight.data - self.embedding.weight.data.detach()) ** 2) 

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if z.ndim == 3:
            z_q = torch.einsum('b h c -> b c h', z_q)
        elif z.ndim == 4:
            z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class VectorQuantize(nn.Module):
    """Local implementation of vector quantization to avoid external dependencies."""
    
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.8,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: float = 0.0,
        threshold_ema_dead_code_ratio: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim if codebook_dim is not None else dim
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.threshold_ema_dead_code_ratio = threshold_ema_dead_code_ratio
        
        # Initialize codebook
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.register_buffer('ema_codebook_value', torch.zeros(self.codebook_size, self.codebook_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(self.codebook_size))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            quantized: Quantized tensor
            indices: Codebook indices
            commit_loss: Commitment loss
        """

        # Permute input to (batch_size, ..., dim) for consistent processing
        original_ndim = x.ndim
        if original_ndim == 3: # (b c h)
            x = x.permute(0, 2, 1).contiguous() # (b h c)
        elif original_ndim == 4: # (b c h w)
            x = x.permute(0, 2, 3, 1).contiguous() # (b h w c)
        
        # Reshape for computation
        input_shape = x.shape
        x_flat = x.reshape(-1, self.dim) # (N, dim) where N = B * H * W or B * L
        
        # Compute distances to codebook (using Euclidean distance)
        # (N, codebook_size)
        distances = torch.cdist(x_flat, self.codebook.weight)
        
        # Find closest codebook entries
        indices = torch.argmin(distances, dim=-1) # (N,)
        
        # Get quantized vectors
        # (N, codebook_dim)
        quantized_flat = self.codebook(indices)
        
        # Straight-through estimator:
        # In the backward pass, gradients from `quantized` bypass the argmin and flow directly to `x_flat`
        # and from `quantized_flat` to `self.codebook.weight` for the reconstruction loss part.
        # However, for the VQ layer itself, we need to explicitly copy gradients.
        quantized = x_flat + (quantized_flat - x_flat).detach() # This is the core of the straight-through estimator

        # Reshape back to original spatial dimensions
        quantized = quantized.reshape(input_shape)
        
        # Compute commitment loss
        # This part encourages `x` to be close to the chosen `quantized` vector.
        # Gradients from this loss only flow to `x` (encoder), not to the `codebook.weight`.
        commit_loss = F.mse_loss(quantized_flat.detach(), x_flat) * self.commitment_weight
        
        # Update codebook (EMA)
        if self.training:
            with torch.no_grad(): # Ensure these updates don't create graph
                # One-hot encode indices for selected codebook entries
                # (N, codebook_size)
                indices_one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
                
                # Sum `x_flat` for each codebook entry used
                # (codebook_size, dim)
                # This accumulates the sum of input vectors that mapped to each codebook entry
                cluster_sum = torch.matmul(indices_one_hot.T, x_flat)
                
                # Sum the counts for each codebook entry
                # (codebook_size,)
                cluster_size = indices_one_hot.sum(dim=0)
                
                # Update EMA for cluster sizes
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size)
                
                # Update EMA for codebook values (sum of vectors mapped to each code)
                self.ema_codebook_value.mul_(self.decay).add_(cluster_sum)

                # Normalize and update the actual codebook weights
                # Apply Laplace smoothing (add 1e-5) to avoid division by zero for unused codes
                # The small epsilon helps when ema_cluster_size is zero.
                # Adding cluster_size.unsqueeze(-1) + 1e-5 to denominator directly
                # gives the average of inputs mapping to each code.
                updated_codebook = self.ema_codebook_value / (self.ema_cluster_size.unsqueeze(-1) + 1e-5)
                
                self.codebook.weight.data.copy_(updated_codebook) # Directly copy the updated values

                # Handle dead codes (optional, but good practice)
                # If a code is rarely used (below threshold_ema_dead_code_ratio),
                # reinitialize it to a random encoder output or a random code.
                # This part is more complex and depends on specific strategies.
                # For simplicity, I'm omitting a full implementation here,
                # but you'd check `ema_cluster_size` and reinitialize `codebook.weight.data`
                # for codes that fall below the threshold.
        
        # Reshape indices back
        indices = indices.reshape(input_shape[:-1]) # (B, H, W) or (B, L)

        # Permute output back to original channel-first format
        if original_ndim == 3: # (b c h)
            quantized = quantized.permute(0, 2, 1).contiguous()
        elif original_ndim == 4: # (b c h w)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, indices, commit_loss



class ResidualVQ(nn.Module):
    """Local implementation of residual vector quantization."""
    
    def __init__(
        self,
        dim: int,
        num_quantizers: int = 1,
        codebook_size: int = 8192,
        decay: float = 0.8,
        commitment_weight: float = 1.0,
        stochastic_sample_codes: bool = False,
        sample_codebook_temp: float = 0.1,
        shared_codebook: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.stochastic_sample_codes = stochastic_sample_codes
        self.sample_codebook_temp = sample_codebook_temp
        self.shared_codebook = shared_codebook
        
        # Create quantizers
        if shared_codebook:
            self.quantizers = nn.ModuleList([
                VectorQuantize(
                    dim=dim,
                    codebook_size=codebook_size,
                    decay=decay,
                    commitment_weight=commitment_weight
                ) for _ in range(num_quantizers)
            ])
        else:
            self.quantizers = nn.ModuleList([
                VectorQuantize(
                    dim=dim,
                    codebook_size=codebook_size,
                    decay=decay,
                    commitment_weight=commitment_weight
                ) for _ in range(num_quantizers)
            ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            quantized: Quantized tensor
            indices: Codebook indices
            commit_loss: Commitment loss
        """
        batch_size, seq_len, dim = x.shape
        residual = x
        quantized = torch.zeros_like(x)
        all_indices = []
        total_commit_loss = 0.0
        
        for i, quantizer in enumerate(self.quantizers):
            # Quantize residual
            q, indices, commit_loss = quantizer(residual)
            
            # Accumulate quantized output
            quantized = quantized + q
            
            # Update residual
            residual = residual - q
            
            # Store indices
            all_indices.append(indices)
            total_commit_loss = total_commit_loss + commit_loss
        
        # Stack indices
        indices = torch.stack(all_indices, dim=-1)  # (batch_size, seq_len, num_quantizers)
        
        return quantized, indices, total_commit_loss 