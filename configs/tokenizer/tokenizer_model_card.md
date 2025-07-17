---
language: en
license: mit
tags:
- physics
- simulation
- vqvae
- tokenizer
- wave-equation
- vorticity
---

# Zebra-VQVAE Tokenizer

## Model Description

Zebra-VQVAE is a Vector Quantized Variational Autoencoder (VQVAE) designed to encode physical states (such as wave equations and vorticity fields) into discrete tokens. This tokenizer is specifically optimized for capturing the spatial patterns and dynamics of physical systems, making it suitable for use with transformer-based models like LLaMA.

## Architecture

The model consists of:
- Encoder: Convolutional neural network that compresses input states
- Codebook: Learnable discrete codebook for vector quantization
- Decoder: Convolutional neural network that reconstructs states from tokens

### Key Parameters

- Number of embeddings: {num_embeddings}
- Embedding dimension: {embedding_dim}
- Commitment cost: {commitment_cost}
- Codebook decay: {decay}

## Training Data

The tokenizer was trained on:
- 2D wave equation simulations
- Vorticity field simulations

## Training Procedure

The model was trained using:
- Reconstruction loss
- Codebook loss
- Perceptual loss (optional)

### Training Hyperparameters

- Learning rate: {learning_rate}
- Batch size: {batch_size}
- Max steps: {max_steps}
- Warmup steps: {warmup_steps}
- Gradient clipping: {gradient_clip_val}
- Gradient accumulation: {accumulate_grad_batches}

## Evaluation

The tokenizer was evaluated on:
- Reconstruction quality (MSE, SSIM)
- Codebook usage statistics
- Token distribution
- Compression ratio

## Usage

```python
from zebra.models.tokenizer.vqvae2d import VQVAE2D

# Load tokenizer
tokenizer = VQVAE2D.from_pretrained("your-username/zebra-tokenizer")

# Encode a physical state
tokens = tokenizer.encode(state)

# Decode tokens back to state
reconstructed_state = tokenizer.decode(tokens)

# Get token embeddings
embeddings = tokenizer.get_embeddings(tokens)
```

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{zebra-vqvae,
  author = {Your Name},
  title = {Zebra-VQVAE: A Tokenizer for Physical Systems},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/your-username/zebra-tokenizer}}
}
```

## Limitations

- The quality of tokenization depends on the training data
- The model may not generalize well to unseen physical systems
- The compression ratio is fixed by the architecture
- The codebook size may need to be tuned for different applications

## License

This model is licensed under the MIT License. 