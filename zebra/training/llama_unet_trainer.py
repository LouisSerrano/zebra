import torch
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing import Dict, Optional, Tuple
from einops import rearrange

class LLaMATrainer(L.LightningModule):
    """PyTorch Lightning trainer for LLaMA model."""
    
    def __init__(
        self,
        model,
        tokenizer,
        training_config
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.tokenizer = tokenizer

        self.automatic_optimization = False
        for k, v in training_config.items():
            setattr(self, k, v)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        # Tokenize input
        if self.need_to_encode:
            images, context_images = batch
            t = images.shape[-1]

            if images.ndim == 4:
                with torch.no_grad():
                    x = rearrange(images, "b c h t -> (b t) c h").float()
                    codes, indices = self.tokenizer(x, return_codes=True)
                    tokens = rearrange(indices, '(b t) h c -> b c h t', t=t) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}

                with torch.no_grad():
                    k = context_images.shape[1]
                    x = rearrange(context_images, "b k c h t -> (b k t) c h").float()
                    codes, indices = self.tokenizer(x, return_codes=True)
                    h_lat = indices.shape[1]
                    context_tokens = rearrange(indices, '(b k t) h c -> b k c h t', t=t, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}
            
            elif images.ndim == 5:
                with torch.no_grad():
                    x = rearrange(images, "b c h w t -> (b t) c h w").float()
                    codes, indices = self.tokenizer(x, return_codes=True)
                    tokens = rearrange(indices, '(b t) h w c -> b c h w t', t=t) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}
                    
                with torch.no_grad():
                    k = context_images.shape[1]
                    x = rearrange(context_images, "b k c h w t -> (b k t) c h w").float()
                    codes, indices = self.tokenizer(x, return_codes=True)
                    h_lat = indices.shape[1]
                    context_tokens = rearrange(indices, '(b k t) h w c -> b k c h w t', t=t, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}
        else:
            tokens, context_tokens = batch
        
        # Forward pass
        outputs = self.model(tokens, context_sequences=context_tokens)
        loss = outputs['loss']

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        sch.step()
        
        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # Tokenize input
        if self.need_to_encode:
            images, context_images = batch
            t = images.shape[-1]

            if images.ndim == 4:
                with torch.no_grad():
                    x = rearrange(images, "b c h t -> (b t) c h").float()
                    codes, indices = self.tokenizer(x, return_codes=True)
                    tokens = rearrange(indices, '(b t) h c -> b c h t', t=t) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}

                with torch.no_grad():
                    k = context_images.shape[1]
                    x = rearrange(context_images, "b k c h t -> (b k t) c h").float()
                    codes, indices = self.tokenizer(x, return_codes=True)
                    h_lat = indices.shape[1]
                    context_tokens = rearrange(indices, '(b k t) h c -> b k c h t', t=t, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}
            
            elif images.ndim == 5:
                with torch.no_grad():
                    x = rearrange(images, "b c h w t -> (b t) c h w").float()
                    codes, indices = self.tokenizer(x, return_codes=True)
                    tokens = rearrange(indices, '(b t) h w c -> b c h w t', t=t) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}
                    
                with torch.no_grad():
                    k = context_images.shape[1]
                    x = rearrange(context_images, "b k c h w t -> (b k t) c h w").float()
                    codes, indices = self.tokenizer(x, return_codes=True)
                    h_lat = indices.shape[1]
                    context_tokens = rearrange(indices, '(b k t) h w c -> b k c h w t', t=t, k=k) # c varies, then h , then temporal ((c1 c2)_1, ..., (c1 c2)_L)_{t=0}, ((c1 c2)_1, ..., (c1 c2)_L)_{t=1}
        else:
            tokens, context_tokens = batch
        
        # Forward pass
        outputs = self.model(tokens, context_sequences=context_tokens)
        loss = outputs['loss']
        
        # Log loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    
    def on_train_epoch_start(self) -> None:
        """Called at the beginning of each training epoch."""
        # Update model's step counter if needed
        if hasattr(self.model, 'step'):
            self.model.step = self.global_step
    
    def on_validation_epoch_start(self) -> None:
        """Called at the beginning of each validation epoch."""
        # Set model to eval mode
        self.model.eval()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        # Set model back to train mode
        self.model.train()
    
    def generate(
        self,
        sequences: torch.Tensor,
        context_sequences: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate sequences using the model."""
        return self.model.generate(sequences, context_sequences, **kwargs) 