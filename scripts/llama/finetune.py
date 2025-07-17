import os
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from transformers import PreTrainedTokenizer

from zebra.training.llama_trainer import LLaMATrainer
from zebra.utils.data import get_data_loader
from zebra.models.tokenizer.vqvae2d import VQVAE2D

def load_pretrained_model(config: DictConfig) -> LLaMATrainer:
    """Load a pretrained model for finetuning."""
    model = LLaMATrainer(config)
    
    # Load pretrained weights
    if config.finetune.pretrained_model_path:
        # Load from local path
        if os.path.exists(config.finetune.pretrained_model_path):
            model.model.load_state_dict(torch.load(config.finetune.pretrained_model_path))
        # Load from Hugging Face Hub
        else:
            model.model.from_pretrained(config.finetune.pretrained_model_path)
    
    # Freeze layers if specified
    if config.finetune.freeze_other_layers:
        for name, param in model.named_parameters():
            if not any(layer in name for layer in config.finetune.trainable_layers):
                param.requires_grad = False
    
    return model

@hydra.main(config_path="../../zebra/configs/llama", config_name="finetune")
def main(config: DictConfig):
    # Set random seed
    pl.seed_everything(42)
    
    # Initialize wandb
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        tags=config.wandb.tags
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_dir, wandb_logger.experiment.name),
        filename=config.checkpoint.filename,
        save_top_k=config.checkpoint.save_top_k,
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        every_n_train_steps=config.checkpoint.every_n_train_steps
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_steps=config.max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches
    )
    
    # Load pretrained model and data loaders
    model = load_pretrained_model(config)
    train_loader, val_loader = get_data_loader(config)
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, wandb_logger.experiment.name, "final_model")
    model.model.save_pretrained(final_model_path)
    
    # Push to Hugging Face Hub if configured
    if config.huggingface.push_to_hub:
        model.model.push_to_hub(
            config.huggingface.repo_name,
            use_auth_token=config.huggingface.token
        )

if __name__ == "__main__":
    main() 