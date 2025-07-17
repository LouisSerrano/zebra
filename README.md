# ZEBRA

<div align="center">
    <img src="images/zebra.png" alt="description" width="50%">
</div>


# Install 
### Editable version of Huggingface Transformers  🤗 (includes Zebra) 🦓
```sh
git clone https://github.com/Purple-PI/transformers.git
cd transformers
pip install -e .
```
###  Datasets
```sh
pip install datasets
```
# TODO Tokenizer
 - [X] Refactor Tokenizer config
 - [X] Tokenizer push to hub
 - [X] Check code
 - [ ] Tokenizer Trainer (ligthning ?)
<<<<<<< HEAD
 - [X] Tokenizer Batch_decode
 - [X] Make sure Codebook properly saved
=======
 - [ ] Tokenizer Batch_decode
 - [ ] Add parameters for normalizing data
 - [ ] Make sure Codebook properly saved
>>>>>>> 52ea8522fb5ee3775f639871df08b0971ce3348e
 - [ ] Push subsets for data
 - [ ] Test all configurations on GPUs
 

# TODO LLM 
 - [x] Implement autoregressive (Zebra) 
 - [x] Training code for Zebra
 - [x] Implement masked language model  (ZebraMLM)
 - [x] Training code for ZebraMLM
 - [ ] Dual vocab space for factorized codebook

### Pipeline
1) Train tokenizer [config/tokenizer.yaml , train/tokenizer_pl.py]
2) Tokenize dataset [config/tokenizedata.yaml,  tokenizer_inference/tokenize_data.py]
3) Train MLM [config/zebra.yaml, train_zebra/train_mlm.py]



### Load Tokenizer and Data

```py
from magvit_1d import Tokenizer, TokenizerConfig
from basic1d_dataset import DataCollator

collator = DataCollator(cfg.dataset.name, cfg.dataset.subset, cfg.dataset.slice_size, cfg.dataset.channels)
dataloader = DataLoader(collator, batch_size=2, shuffle=True, num_workers=4)
# instantiate Tokenizer
config_tokenizer = TokenizerConfig(**cfg.tokenizer)
tokenizer = Tokenizer(config_tokenizer)
#train script
...
#
# Tokenize Raw Frames
tensor = torch.randn(4, 256, 17, 1)
# Make batch of tokens 
# Fomat [mu1, sig1, f1]
batch = tokenizer.batch_encode(tensor)
# [4 , 476]
# tensor([[ 19,   5,   0,  ...,  74, 263,  84],
#         [ 20,   4,   4,  ..., 141, 263, 294],
#         [ 19,   6,   0,  ..., 263, 141, 263],
#         [ 20,   3,   0,  ..., 294, 263, 294]])
#save model

config_tokenizer.push_to_hub('name/model_name')
tokenizer = Tokenizer(config_tokenizer).push_to_hub('name/model_name')

# load trained model
config_tokenizer = TokenizerConfig.from_pretrained('name/model_name')
tokenizer = Tokenizer.from_pretrained('name/model_name', config = config_tokenizer)

# Tokenize data
codes, index = tokenizer.batch_encode(frames) 
# Decode data
frames = tokenizer.batch_decode(index) # index = (List[List[int]])
```
### Load Language Models

```py
from transformers import ZebraConfig, ZebraForCausalLM
#define config (see config file)
config = ZebraConfig(**cfg.model)
#instantiate the model
model = ZebraForCausalLM(config)
#train script
...
#
#save model
model.push_to_hub('name/model_name', private = True)
# load model trained model
model = ZebraForCausalLM.from_pretrained('name/model_name')

```

## Available Dataset
### 1D PDE from PDEBench

360K Raw data downsampled to 256,101 (20.0 GB compressed):
 - Advection
 - Burgers
 - Reaction-diffusion

```py
dataset = datasets.load_dataset('erbacher/PDEBench-1D')
# load only CFD_Rand_Eta0.01_Zeta0.01_periodic_Train
ds1 = datasets.load_dataset('erbacher/PDEBench-1D', 'CFD_Rand_Eta0.01_Zeta0.01_periodic')
ds2 = datasets.load_dataset('erbacher/PDEBench-1D', 'ReacDiff_Nu0.5_Rho10.0')

```

480K Raw data downsampled to 256,101   (34.4 GB compressed):

 - Advection
 - Burgers
 - Reaction-diffusion
 - Compressible Navier Stokes Pressure channel
 - Compressible Navier Stokes density channel
 - Compressible Navier Stokes X velocity channel
```py
dataset = datasets.load_dataset('erbacher/PDEBench-1D-full')

```
#### Additional resources to process datasets: https://huggingface.co/docs/datasets/process





<h2>Contributors</h2>

- Louis Serrano

- Etienne Lenaour

- Pierre Erbacher
   
<h2>References</h2>

<p> <em>PDEBench</em><a> https://arxiv.org/abs/2210.07182 </a>

