import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from zebra.models.tokenizer.vqvae1d import VQVAE1D
from zebra.models.tokenizer.vqvae2d import VQVAE2D
from zebra.training.tokenizer_trainer import TokenizerTrainer
from zebra.utils.data import get_data, load_wave2d, load_vort, TemporalDataset


def get_wandb_run_name(cfg: DictConfig) -> str:
    """Create a descriptive wandb run name for tokenizer training."""
    # Get dataset name from path
    dataset_name = cfg.data.dataset_name
    
    # Get model parameters
    model_params = [
        f"emb{cfg.model.codebook_size}",
        f"dim{cfg.model.code_dim}"
    ]
    
    # Combine all parts
    return f"tokenizer_{dataset_name}_{'_'.join(model_params)}"

@hydra.main(config_path="../../configs/tokenizer", config_name="vqvae1d.yaml")
def train(cfg: DictConfig):
    # Set up logging
    print(cfg.training)
    #L.seed_everything(cfg.training.seed)
    
    # Create logger with descriptive run name
    run_name=get_wandb_run_name(cfg)
    logger = WandbLogger(
        project=cfg.logging.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    dataset_name = cfg.data.dataset_name

    # model class
    if dataset_name in ['wave2d', 'vorticity']:
       model_class = VQVAE2D
    else:
        model_class = VQVAE1D

    # load data
    if dataset_name=="wave2d":
        train_loader, val_loader, test_loader = load_wave2d(cfg.data.data_dir, cfg.training.batch_size, cfg.training.batch_size)
    elif dataset_name=="vorticity":
        train_loader, val_loader, test_loader = load_vort(cfg.data.data_dir, cfg.training.batch_size, cfg.training.batch_size)
    else:
        u_train, u_val, u_test = get_data(cfg.data.data_dir, dataset_name, return_params=False)
        train_dataset = TemporalDataset(u_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size)
        val_dataset = TemporalDataset(u_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )

    # Create model
    model = model_class(**cfg.model)

    # Create trainer
    trainer = TokenizerTrainer(
        model=model,
        training_config=cfg.training
    )

    # Create callbacks
    print(cfg.logging.output_dir)
    print(run_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.logging.output_dir, run_name),
            filename="{step}-{val_rel_loss:.2f}",
            monitor="val_rel_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Create Lightning trainer
    pl_trainer = L.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        #gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        check_val_every_n_epoch=10,
        callbacks=callbacks,
        logger=logger,
    )

    # Train
    pl_trainer.fit(
        trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save final model
    if cfg.huggingface.push_to_hub:
        trainer.model.push_to_hub(
            repo_name=cfg.huggingface.repo_name,
            private=cfg.huggingface.private,
            commit_message=cfg.huggingface.commit_message,
            model_card=cfg.huggingface.model_card,
            model_card_template=cfg.huggingface.model_card_template,
        )

if __name__ == "__main__":
    train() 