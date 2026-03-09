


# 1. Official Code
Official PyTorch implementation of Zebra | [Accepted at ICML 2025](https://openreview.net/forum?id=22kNOkkokU))


<p float="center">
  <img src="assets/poster_zebra_v2.jpg" width="800"/>
</p>

To cite our work:

```
inproceedings{
serrano2025zebra,
title={Zebra: In-Context Generative Pretraining for Solving Parametric {PDE}s},
author={Louis Serrano and Armand Kassa{\"\i} Koupa{\"\i} and Thomas X Wang and Pierre ERBACHER and Patrick Gallinari},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=22kNOkkokU}
}
```

# 1. Code installation and setup
## zebra installation
```
conda create -n zebra python=3.9.0
pip install -e .
```

## setup wandb config example

add to your `~/.bashrc`
```
export WANDB_API_TOKEN=your_key
export WANDB_DIR=your_dir
export WANDB_CACHE_DIR=your_cache_dir
export MINICONDA_PATH=your_anaconda_path
```

# 2. Data

All datasets are hosted on [HuggingFace](https://huggingface.co/sogeeking). You can download them using the provided script:

```bash
pip install huggingface_hub

# Download specific datasets
python download_data/download_data_hugging_face.py --datasets vorticity wave gs

# Download a dataset and its OOD counterpart
python download_data/download_data_hugging_face.py --datasets vorticity vorticity_ood

# Download all datasets
python download_data/download_data_hugging_face.py --datasets all

# Specify a custom output directory (default: ./data)
python download_data/download_data_hugging_face.py --datasets vorticity --data_dir /path/to/data
```

Available datasets:

| Dataset | HuggingFace repo | Description |
|---|---|---|
| `vorticity` | [sogeeking/vorticity](https://huggingface.co/datasets/sogeeking/vorticity) | 2D Navier-Stokes (vorticity form) |
| `vorticity_ood` | [sogeeking/vorticity_ood](https://huggingface.co/datasets/sogeeking/vorticity_ood) | OOD evaluation for vorticity |
| `wave` | [sogeeking/wave](https://huggingface.co/datasets/sogeeking/wave) | 2D wave equation |
| `wave_ood` | [sogeeking/wave_ood](https://huggingface.co/datasets/sogeeking/wave_ood) | OOD evaluation for wave |
| `gs` | [sogeeking/gs](https://huggingface.co/datasets/sogeeking/gs) | 2D Gray-Scott reaction-diffusion |
| `gs_ood` | [sogeeking/gs_ood](https://huggingface.co/datasets/sogeeking/gs_ood) | OOD evaluation for Gray-Scott |
| `combined_equation` | [sogeeking/combined-equation-2](https://huggingface.co/datasets/sogeeking/combined-equation-2) | 1D combined equation |
| `advection_diffusion` | [sogeeking/advection-diffusion](https://huggingface.co/datasets/sogeeking/advection-diffusion) | 1D advection-diffusion |
| `heat_nu_forcing2` | [sogeeking/heat-nu-forcing-2](https://huggingface.co/datasets/sogeeking/heat-nu-forcing-2) | 1D heat (varying viscosity & forcing) |
| `burgers_nu_forcing2` | [sogeeking/burgers-nu-forcing-2](https://huggingface.co/datasets/sogeeking/burgers-nu-forcing-2) | 1D Burgers (varying viscosity & forcing) |

# 3. Run experiments 

The code runs only on GPU. We provide sbatch configuration files to run the training scripts. They are located in `bash` and are organized by datasets.
We expect the user to have wandb installed in its environment for monitoring. 
In Zebra, the first step is to launch an tokenizer.py training, in order to learn a finite vocabulary of physical phenomena. The weights of the tokenizer model are automatically saved under its `run_name`.
For the second step, i.e. for training the language model with an in-context pretraining, we need to use the previous `run_name` as input to the config file to load the tokenizer model. The `run_name` can be set in the config file, but can also be generated randomly by default with wandb.

For instance, for `advection` we need to first train the VQVAE:
`sbatch bash/burgers/tokenizer.sh`
and then once we specified the correct run_name in the config:
`sbatch bash/burgers/llama.sh`

# Acknowledgements

This project would not have been possible without these awesome repositories:
* Transformer implementation from hugging face: https://github.com/huggingface/transformers
* MAGVIT implementation from lucidrains: https://github.com/lucidrains/magvit2-pytorch
* PDE Arena : https://github.com/pdearena/pdearena











