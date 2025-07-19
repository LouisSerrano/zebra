import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from .residual_vq import ResidualVQ
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List
from omegaconf import ListConfig
from torch import nn

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

class RelativeL2(nn.Module):
    def forward(self, x, y):
        x = rearrange(x, "b ... -> b (...)")
        y = rearrange(y, "b ... -> b (...)")
        diff_norms = torch.linalg.norm(x - y, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        return (diff_norms / y_norms).mean()


class SpatialUpsample2(Module):
    """Upsamples spatial dimensions by a factor of 2 using pixel shuffle."""
    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1, bias=False)

        self.net = nn.Sequential(
            conv, Rearrange("b (c p1 p2) h w -> b c (h p1) (p2 w)", p1=2, p2=2)
        )
        self.init_conv_(conv)

    def init_conv_(self, conv: nn.Conv2d):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    

class SpatialDownsample2(Module):
    """Downsamples spatial dimensions by a factor of 2 using strided convolution."""
    def __init__(self, dim: int, dim_out: Optional[int] = None, kernel_size: int = 3):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.conv = nn.Conv2d(
            dim, dim_out, kernel_size, stride=2, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class BlurPool1D(nn.Module):
    """A layer to do channel-wise blurring + subsampling on 1D inputs in PyTorch."""

    def __init__(self, filter_size=4, stride=2, padding='same'):
        super(BlurPool1D, self).__init__()
        if filter_size not in [3, 4, 5, 6, 7]:
            raise ValueError('Only filter_size of 3, 4, 5, 6, or 7 is supported.')

        filter_vals = {3: [1., 2., 1.],
                       4: [1., 3., 3., 1.],
                       5: [1., 4., 6., 4., 1.],
                       6: [1., 5., 10., 10., 5., 1.],
                       7: [1., 6., 15., 20., 15., 6., 1.]}

        self.filter = torch.tensor(filter_vals[filter_size], dtype=torch.float32)
        self.filter /= self.filter.sum()
        self.filter = self.filter[None, None, :]

        self.stride = stride
        self.padding = "same"
        #self.padding = padding if padding == 'same' else 0
        #self.padding = max((x.size(2) - 1) * stride + kernel_size - x.size(2), 0) // 2

    def forward(self, x):
        length = x.shape[2]
        if self.padding == 'same':
            pad_size = (self.filter.shape[-1] - 1) // 2
            x = F.pad(x, (pad_size, pad_size), mode='constant')
        channel_num = x.shape[1]
        #print('self.filter', self.filter.shape)
        filter = self.filter.repeat(channel_num, 1, 1).to(x.device)
        return F.conv1d(x, filter, stride=self.stride, groups=channel_num)#padding=self.padding)


class ResBlock(nn.Module):
    """Basic residual block with optional normalization and activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 8,
        pad_mode: str = "zeros",
        norm_fn: Optional[Module] = nn.GroupNorm,
        activation_fn: Module = nn.SiLU,
    ):
        super().__init__()
        self.norm1 = norm_fn(num_groups, in_channels) if norm_fn is not None else nn.Identity()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=False,
        )
        self.norm2 = norm_fn(num_groups, out_channels) if norm_fn is not None else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=False,
        )
        self.activation_fn = activation_fn()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                padding_mode=pad_mode,
                bias=False,
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        
        if hasattr(self, 'shortcut'):
            residual = self.shortcut(residual)
            
        return x + residual

class CausalConv1d(Module):
    @beartype
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int]],
        pad_mode="constant",
        **kwargs,
    ):
        super().__init__()
        ##print('kernel-size', kernel_size)
        # kernel_size = cast_tuple(kernel_size, 1)

        time_kernel_size = kernel_size

        assert is_odd(time_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)

        self.time_pad = time_pad
        self.time_causal_padding = (time_pad, 0)

        # stride = (stride, 1)
        # dilation = (dilation, 1)
        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, bias=False, **kwargs
        )
        # print('conv1d', self.conv.weight.shape, self.conv)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


class ResBlockTime(nn.Module):
    """Basic Residual Block, adapted from magvit1"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="constant",
        norm_fn=nn.GroupNorm, #None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super(ResBlockTime, self).__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.norm1 = norm_fn(num_groups, in_channels) if norm_fn is not None else nn.Identity()
        self.conv1 = CausalConv1d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  pad_mode=pad_mode,)
        
        self.norm2 = norm_fn(num_groups, out_channels) if norm_fn is not None else nn.Identity()
        self.conv2 = CausalConv1d(out_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  pad_mode=pad_mode,)
        
        self.activation_fn = activation_fn()
      
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
       
        return x + residual

class ResBlockDown(nn.Module):
    """Basic Residual Block, adapted from magvit1"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        pad_mode="zeros",
        activation_fn=nn.LeakyReLU,
        norm_fn = nn.GroupNorm, #None,
        use_conv_shortcut=False,
        num_groups=8
    ):
        super(ResBlockDown, self).__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=True,
        )
        self.norm1 = norm_fn(num_groups, out_channels) if norm_fn is not None else nn.Identity()
        self.norm2 = norm_fn(num_groups, out_channels) if norm_fn is not None else nn.Identity()
        self.activation_fn = activation_fn()

        self.blur = BlurPool1D()

        self.shortcut = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            padding_mode=pad_mode,
            bias=True,
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        #print("before blur", x.mean(), x.std())
        x = self.blur(x)
        #print("after blur", x.mean(), x.std())
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        if self.use_conv_shortcut or residual.shape != x.shape:
            residual = self.shortcut(self.blur(residual))
        #print('after residual', )
        return x + residual


class FinalBlock(nn.Module):
    """Basic Residual Block, adapted from magvit1"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=nn.GroupNorm,
        activation_fn=nn.SiLU,
    ):
        super(FinalBlock, self).__init__()
        self.norm = norm_fn(num_groups, in_channels) if norm_fn is not None else nn.Identity() #norm_fn(num_groups, in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
        )
        self.activation_fn = activation_fn()

    def forward(self, x):
        x = self.norm(x)
        x = self.activation_fn(x)
        x = self.conv(x)
        return x
# helper classes

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)


class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # print('Residual', x.shape)
        return self.fn(x, **kwargs) + x


class VQVAE2D(Module):
    """A spatial tokenizer that encodes images into discrete tokens using residual VQ.
    
    The architecture consists of:
    1. An encoder with residual blocks and spatial downsampling
    2. A vector quantizer (ResidualVQ)
    3. A decoder with residual blocks and spatial upsampling
    """
    @beartype
    def __init__(
        self,
        channels: int = 1,
        init_dim: int = 64,
        max_dim: int = 256,
        layers: Union[List[str], ListConfig[str]] = (
            "residual",
            "compress_space",
            "compress_space",
            "compress_space",
            "compress_space",
            "residual",
        ),
        code_dim: int = 128,
        num_codebooks: int = 1,
        codebook_size: int = 8192,
        num_groups: int = 8,
        pad_mode: str = "circular",
        input_conv_kernel_size: int = 5,
        residual_conv_kernel_size: int = 3,
        quantization_type: str = "resvq",
        shared_codebook: bool = True,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.channels = channels
        self.code_dim = code_dim
        self.num_groups = num_groups
        self.commitment_weight = commitment_weight
        self.rel_loss = RelativeL2()
        self.quantization_type = quantization_type
        self.commitment_weight = commitment_weight
        self.codebook_size = codebook_size
        # Build encoder
        self.encoder_layers = ModuleList([])
        self.decoder_layers = ModuleList([])

        # self.conv_out = nn.Conv1d(init_dim, channels, output_conv_kernel_size, padding=output_conv_kernel_size // 2, padding_mode=pad_mode)

        self.init_dim = init_dim
        dim = init_dim
        dim_out = dim

        has_cond = False
    
        for layer_def in layers:
            layer_type, *layer_params = cast_tuple(layer_def)

            if layer_type == "residual":
                encoder_layer = ResBlock(
                    dim,
                    dim,
                    residual_conv_kernel_size,
                    num_groups=self.num_groups,
                    pad_mode=pad_mode,
                )
                decoder_layer = ResBlock(
                    dim,
                    dim,
                    residual_conv_kernel_size,
                    num_groups=self.num_groups,
                    pad_mode=pad_mode,
                )
            
            elif layer_type == "causal_time":
                encoder_layer = ResBlockTime(
                    dim,
                    dim,
                    residual_conv_kernel_size,
                    num_groups=self.num_groups,
                )
                decoder_layer = ResBlockTime(
                    dim,
                    dim,
                    residual_conv_kernel_size,
                    num_groups=self.num_groups,
                )

            elif layer_type == "compress_space":
                dim_out = dim*2
                dim_out = min(dim_out, max_dim)

                encoder_layer = nn.Sequential(SpatialDownsample2(
                    dim, dim
                ), ResBlock(dim, 
                            dim_out, 
                            residual_conv_kernel_size,
                            num_groups=self.num_groups,
                            pad_mode=pad_mode,))
                
                decoder_layer = nn.Sequential(SpatialUpsample2(dim_out, dim_out),
                                              ResBlock(dim_out, 
                                                dim, 
                                                residual_conv_kernel_size,
                                                num_groups=self.num_groups,
                                                pad_mode=pad_mode,))                                              

            else:
                raise ValueError(f"unknown layer type {layer_type}")

            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)
            dim = dim_out

        code_dim = self.code_dim
        self.embedding_dim = code_dim

        self.decoder_layers.insert(
            0,
            nn.Conv2d(
                self.embedding_dim if not self.quantization_type in ["softvq","fsq"] else dim,
                dim,
                kernel_size=residual_conv_kernel_size,
                padding=residual_conv_kernel_size // 2,
                padding_mode=pad_mode,
            ),
        )
        self.encoder_layers.insert(
            0,
            nn.Conv2d(
                self.channels,
                self.init_dim,
                kernel_size=input_conv_kernel_size,
                bias=False,
                padding=input_conv_kernel_size // 2,
                padding_mode=pad_mode,
            ),
        )

        self.encoder_layers.append(
            FinalBlock(
                dim,
                self.embedding_dim if not self.quantization_type in ["softvq","fsq"] else dim,
                kernel_size=1,
                pad_mode=pad_mode,
                num_groups=self.num_groups,
                norm_fn=nn.GroupNorm if self.quantization_type == "lfq" else None,
            )
        )
        self.decoder_layers.append(
            FinalBlock(
                self.init_dim,
                self.channels,
                kernel_size=input_conv_kernel_size,
                pad_mode=pad_mode,
                num_groups=self.num_groups,
                norm_fn=nn.GroupNorm if self.quantization_type == "lfq" else None,
            )
        )
        self.quantizers = ResidualVQ(
                        dim = code_dim,
                        num_quantizers = num_codebooks,      # specify number of quantizers
                        codebook_size = codebook_size,    # codebook size
                        stochastic_sample_codes = True, # True
                        sample_codebook_temp = 0.1,         # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
                        shared_codebook = shared_codebook   
                        )

    def _build_encoder_blocks(self, init_dim: int, max_dim: int, pad_mode: str) -> ModuleList:
        """Build encoder blocks with residual blocks and spatial downsampling."""
        blocks = ModuleList([])
        dim = init_dim
        
        while dim < max_dim:
            dim_out = min(dim * 2, max_dim)
            blocks.extend([
                SpatialDownsample2(dim, dim, antialias=True),
                ResBlock(dim, dim_out, num_groups=self.num_groups, pad_mode=pad_mode)
            ])
            dim = dim_out
            
        return blocks

    def _build_decoder_blocks(self, max_dim: int, init_dim: int, pad_mode: str) -> ModuleList:
        """Build decoder blocks with residual blocks and spatial upsampling."""
        blocks = ModuleList([])
        dim = max_dim
        
        while dim > init_dim:
            dim_out = max(dim // 2, init_dim)
            blocks.extend([
                ResBlock(dim, dim_out, num_groups=self.num_groups, pad_mode=pad_mode),
                SpatialUpsample2(dim_out, dim_out)
            ])
            dim = dim_out
            
        return blocks

    @beartype
    def encode(self, x: Tensor, quantize: bool = False) -> Tensor:
        """Encode input images through the encoder and optionally quantize."""
        for layer in self.encoder_layers:
            x = layer(x)
            
        if quantize:
            b, c, h, w = x.shape
            x = rearrange(x, "b c h w -> b (h w) c")
            x, _, _ = self.quantizers(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=h)
            
        return x

    @beartype
    def decode(self, x: Tensor) -> Tensor:
        """Decode quantized codes back to image space."""
        for layer in self.decoder_layers:
            x = layer(x)
        return x

    @beartype
    def forward(
        self,
        x: Tensor,
        return_loss: bool = False,
        return_codes: bool = False,
        return_recon: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, dict], Tuple[Tensor, Tensor]]:
        """Forward pass through the tokenizer.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            return_loss: Whether to return the reconstruction loss
            return_codes: Whether to return the quantized codes
            return_recon: Whether to return the reconstructed images
            
        Returns:
            If return_loss is True: (loss, loss_dict, reconstruction)
            If return_codes is True: (quantized, indices)
            If return_recon is True: reconstruction
            Otherwise: reconstruction
        """
        assert (return_loss + return_codes + return_recon) <= 1


        inp = x.clone()

        # Encode and quantize
        x = self.encode(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        quantized, indices, commit_loss = self.quantizers(x)
        commit_loss = commit_loss.mean()
        quantized = rearrange(quantized, "b (h w) c -> b c h w", h=h)
        indices = rearrange(indices, "b (h w) c -> b c h w", h=h)
        aux_losses = commit_loss * self.commitment_weight

        if return_codes and not return_recon:
            return quantized, indices

        # Decode
        recon = self.decode(quantized)

        if not return_loss:
            return recon

         # Compute losses
        rel_loss = self.rel_loss(recon, inp)
        loss = rel_loss + commit_loss * self.commitment_weight

        return recon, {"loss": loss, "rel_loss": rel_loss, "commit_loss": commit_loss}