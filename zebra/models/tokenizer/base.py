from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class BaseTokenizer(nn.Module, ABC):
    """Base class for all tokenizers in the Zebra library."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Encode input tensor into discrete tokens.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width, time]
            
        Returns:
            Tuple containing:
            - Encoded tokens
            - Dictionary with additional information (e.g., loss breakdown)
        """
        pass
    
    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens back into the original space.
        
        Args:
            tokens: Input tokens
            
        Returns:
            Decoded tensor
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Forward pass through the tokenizer.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing:
            - Total loss
            - Loss breakdown dictionary
            - Reconstructed output
        """
        pass
    
    def get_last_layer(self) -> Optional[nn.Module]:
        """Get the last layer of the model for feature extraction.
        
        Returns:
            The last layer module or None if not applicable
        """
        return None 