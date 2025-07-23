import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedModel
from einops import rearrange
import os
import json

class Zebra(PreTrainedModel):
    """LLaMA model for sequence modeling with tokenizer integration."""
    
    config_class = LlamaConfig
    base_model_prefix = "zebra"
    
    def __init__(self, config: Dict):
        super().__init__(LlamaConfig(**config))
        self.config = config
        
        # Initialize LLaMA model
        llama_config = LlamaConfig(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            num_hidden_layers=config['num_hidden_layers'],
            num_attention_heads=config['num_attention_heads'],
            intermediate_size=config['intermediate_size'],
            max_position_embeddings=config['max_position_embeddings'],
            rms_norm_eps=config.get('rms_norm_eps', 1e-6),
            pad_token_id=config['pad_token_id'],
            bos_token_id=config['bos_token_id'],
            eos_token_id=config['eos_token_id'],
        )
        self.model = LlamaForCausalLM(llama_config)
        
        # Special tokens
        self.num_dimensions = config['num_dimensions']
        self.max_length = config['max_length']

        self.bos_token_id = config['bos_token_id']
        self.eos_token_id = config['eos_token_id']
        self.context_token_id = config['context_token_id']
        self.input_token_id = config['input_token_id']
        self.target_token_id = config['target_token_id']
        self.bot_token_id = config['bot_token_id']
        self.eot_token_id = config['eot_token_id']
        self.pad_token_id = config['pad_token_id']
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained model from a local path or the Hugging Face Hub."""
        config = kwargs.pop("config", None)
        if config is None:
            config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        
        model = cls(config.to_dict())
        model.model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        self.model.save_pretrained(save_directory, **kwargs)
        
        # Save special tokens
        special_tokens = {
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'context_token_id': self.context_token_id,
            'input_token_id': self.input_token_id,
            'target_token_id': self.target_token_id,
            'bot_token_id': self.bot_token_id,
            'eot_token_id': self.eot_token_id,
            'pad_token_id': self.pad_token_id,
        }
        
        with open(os.path.join(save_directory, 'special_tokens.json'), 'w') as f:
            json.dump(special_tokens, f)
    
    def prepare_sequence(
        self,
        sequences: torch.Tensor,
        context_sequences: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input sequences for the model.
        
        Args:
            sequences: Input sequences of shape [batch_size, channels, height, width, time]
            context_sequences: Optional context sequences of shape [batch_size, num_context, channels, height, width, time]
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (input_ids, labels)
        """
        b = sequences.shape[0]
        device = sequences.device

        #print('sequences.shape', sequences.shape)
        #print('context_sequences.shape', context_sequences.shape)

        if sequences.ndim==4:
            num_dimensions=1
        elif sequences.ndim==5:
            num_dimensions=2
      
        # Reshape sequences
        if num_dimensions==1: 
            sequences = rearrange(sequences, 'b c h t -> b (t h c)')
        elif num_dimensions==2: 
            sequences = rearrange(sequences, 'b c h w t -> b (t h w c)')
        
        if context_sequences is not None:
            if num_dimensions==1: 
                context_sequences = rearrange(context_sequences, 'b k c h t -> b k t (h c)')
            elif num_dimensions==2:
                context_sequences = rearrange(context_sequences, 'b k c h w t -> b k t (h w c)')

        # Prepare sequences using single torch.cat operation
        if context_sequences is not None:
            # Create all token tensors
            bos_token = torch.full((b, 1), self.bos_token_id, device=sequences.device)
            bot_token = torch.full((b, 1), self.bot_token_id, device=sequences.device)
            eot_token = torch.full((b, 1), self.eot_token_id, device=sequences.device)
            eos_token = torch.full((b, 1), self.eos_token_id, device=sequences.device)
            
            # Build context parts
            context_parts = []
            for k in range(context_sequences.shape[1]):
                context_parts.append(torch.cat([bot_token, rearrange(context_sequences[:, k], 'b t h -> b (t h)'), eot_token], dim=1))
            context_parts = torch.cat(context_parts, dim=1)
            
            # Concatenate everything at once
            input_ids = torch.cat([
                bos_token,
                context_parts,
                bot_token, sequences, eot_token, eos_token
            ], dim=1)
            
            #prepared_sequences = input_ids.tolist()
        else:
            # Simpler case without context sequences
            bos_token = torch.full((b, 1), self.bos_token_id, device=sequences.device)
            bot_token = torch.full((b, 1), self.bot_token_id, device=sequences.device)
            eot_token = torch.full((b, 1), self.eot_token_id, device=sequences.device)
            eos_token = torch.full((b, 1), self.eos_token_id, device=sequences.device)
            
            input_ids = torch.cat([
                bos_token, bot_token, sequences, eot_token, eos_token
            ], dim=1)
            
            #prepared_sequences = input_ids.tolist()

        labels = input_ids.clone()
        
        return input_ids[:, :self.max_length], labels[:, :self.max_length]
    
    def forward(
        self,
        sequences: torch.Tensor,
        context_sequences: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict:
        """Forward pass through the model.
        
        Args:
            sequences: Input sequences
            context_sequences: Optional context sequences
            return_loss: Whether to compute and return the loss
            
        Returns:
            Dictionary containing model outputs
        """
        # Prepare sequences
        input_ids, labels = self.prepare_sequence(sequences, context_sequences)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            labels=labels if return_loss else None,
            return_dict=True
        )
        
        return {
            'loss': outputs.loss if return_loss else None,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }
    
    def generate(
        self,
        sequences: torch.Tensor,
        context_sequences: Optional[torch.Tensor] = None,
        max_length: int = 8192,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """Generate sequences using the model.
        
        Args:
            sequences: Input sequences
            context_sequences: Optional context sequences
            max_length: Maximum sequence length
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated sequences
        """
        # Prepare sequences
        input_ids, _ = self.prepare_sequence(sequences, context_sequences)
        
        # Generate
        generated = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        return generated 
