import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb
from transformers import PreTrainedTokenizer

from zebra.models.llama.model import Zebra

from zebra.training.tokenizer_trainer import TokenizerTrainer
from zebra.training.llama_trainer import LLaMATrainer
from zebra.utils.data import get_data, load_wave2d, load_vort, TemporalDataset, TemporalDatasetWithContext, tokenize_dataset

def get_wandb_run_name(cfg: DictConfig) -> str:
    """Create a descriptive wandb run name for LLaMA pretraining."""
    # Get dataset name from path
    dataset_name = cfg.data.dataset_name
    
    # Get model parameters
    model_params = [
        f"dim{cfg.model.hidden_size}",
        f"layers{cfg.model.num_hidden_layers}",
        f"heads{cfg.model.num_attention_heads}",
        f"ctx{cfg.data.num_context_trajectories}",
    ]
    # Add training parameters
    train_params = [
        f"bs{cfg.training.batch_size}",
        f"lr{cfg.training.learning_rate}",
    ]
    # Combine all parts
    return f"llama_{dataset_name}_{'_'.join(model_params)}_{'_'.join(train_params)}"

@hydra.main(config_path="../../configs/llama", config_name="base")
def train(cfg: DictConfig):
    # Set up logging
    L.seed_everything(cfg.training.seed)
    
    # Create logger with descriptive run name
    run_name = get_wandb_run_name(cfg)
    logger = WandbLogger(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        name=get_wandb_run_name(cfg),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    dataset_name = cfg.data.dataset_name

    tokenizer = TokenizerTrainer.load_from_checkpoint(cfg.data.tokenizer_path)
    tokenizer = tokenizer.model.eval()
    tokenizer.to("cuda") if cfg.training.devices>0 else tokenizer.to("cpu")

     # load data
    if dataset_name=="wave2d":
        if not cfg.training.tokenize_on_the_fly:
            train_loader, val_loader, test_loader = load_wave2d(cfg.data.data_dir, cfg.training.batch_size, cfg.training.batch_size, sub_t=1, slice_size=30)
            token_train, token_val, token_test = tokenize_dataset(cfg.data.token_dataset_path, run_name, train_loader, val_loader, test_loader, tokenizer, device=torch.device("cuda") if cfg.training.devices>0 else torch.device("cpu"))
            train_dataset = TemporalDatasetWithContext(token_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            val_dataset = TemporalDatasetWithContext(token_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            test_dataset = TemporalDatasetWithContext(token_test, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)

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

    elif dataset_name=="vorticity":
        if not cfg.training.tokenize_on_the_fly:
            train_loader, val_loader, test_loader = load_vort(cfg.data.data_dir, cfg.training.batch_size, cfg.training.batch_size, sub_t=1, slice_size=30)
            token_train, token_val, token_test = tokenize_dataset(cfg.data.token_dataset_path, run_name, train_loader, val_loader, test_loader, tokenizer, device=torch.device("cuda") if cfg.training.devices>0 else torch.device("cpu"))
            train_dataset = TemporalDatasetWithContext(token_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            val_dataset = TemporalDatasetWithContext(token_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            test_dataset = TemporalDatasetWithContext(token_test, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)

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
    else:
        u_train, u_val, u_test = get_data(cfg.data.data_dir, dataset_name, return_params=False)

        if not cfg.training.tokenize_on_the_fly:
            train_loader = torch.utils.data.DataLoader(
                u_train,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                u_val,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            test_loader = torch.utils.data.DataLoader(
                u_test,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            token_train, token_val, token_test = tokenize_dataset(cfg.data.token_dataset_path, run_name, train_loader, val_loader, test_loader, tokenizer, device=torch.device("cuda") if cfg.training.devices>0 else torch.device("cpu"))
            print("token_train.shape", token_train.shape)
            train_dataset = TemporalDatasetWithContext(token_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            val_dataset = TemporalDatasetWithContext(token_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            test_dataset = TemporalDatasetWithContext(token_test, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)

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

        else:
            train_dataset = TemporalDatasetWithContext(u_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            val_dataset = TemporalDatasetWithContext(u_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            test_dataset = TemporalDatasetWithContext(u_test, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)

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

    try:
        codebook_size = tokenizer.codebook_size
    except:
        codebook_size = tokenizer.quantizers.codebook_size
        
    cfg.model.vocab_size = codebook_size+ 8
    cfg.model.bos_token_id = codebook_size 
    cfg.model.eos_token_id = codebook_size + 1
    cfg.model.context_token_id = codebook_size + 2
    cfg.model.input_token_id = codebook_size + 3 
    cfg.model.target_token_id = codebook_size + 4
    cfg.model.bot_token_id = codebook_size + 5 
    cfg.model.eot_token_id = codebook_size + 6 
    cfg.model.pad_token_id = codebook_size + 7 

    model = Zebra(cfg.model)

    # Create trainer
    trainer = LLaMATrainer(
        model=model,
        tokenizer=tokenizer,
        training_config=cfg.training
    )

    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.logging.output_dir, run_name),
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
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
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
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
