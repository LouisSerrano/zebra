import torch
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange
from torch import nn
from time import time

class TokenizerTrainer(L.LightningModule):
    def __init__(
        self,
        model,
        training_config,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.automatic_optimization = False
        for k, v in training_config.items():
            setattr(self, k, v)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        opt = self.optimizers()
        sch = self.lr_schedulers()
        x = batch
        if x.ndim == 4:
            x = rearrange(x, 'b c h t -> (b t) c h')
        elif x.ndim == 5:
            x = rearrange(x, 'b c h w t -> (b t) c h w')

        out, loss_dict = self.model(x, return_loss=True)
        #loss = loss_dict['rel_loss'] + loss_dict['vqloss'] + loss_dict['commit_loss'] + loss_dict['entropy_loss']
        loss = loss_dict["loss"]
        
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        sch.step()
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        self.model.eval()
        x = batch
        if x.ndim == 4:
            x = rearrange(x, 'b c h t -> (b t) c h')
        elif x.ndim == 5:
            x = rearrange(x, 'b c h w t -> (b t) c h w')
        out, loss_dict = self.model(x, return_loss=True)
        #loss = loss_dict['rel_loss'] + loss_dict['vqloss'] + loss_dict['commit_loss']
        loss = loss_dict["loss"]
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f"val_{name}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):

        #optimizer = torch.optim.AdamW([
        #    {'params': self.model.quantizers.parameters(), 'lr': 1e-1, 'weight_decay':0},  # higher LR for VQ
        #    {'params': [p for n, p in self.model.named_parameters() if not n.startswith('quantizers.')], 'lr': self.learning_rate, 'weight_decay':self.weight_decay}
        #])
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_steps,
            eta_min=0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        } 