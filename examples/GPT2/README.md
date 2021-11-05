# Opacus Compatible GPT-2

To the best of our knowledge, this is the first Pytorch implementation of GPT-2
that is compatible with Opacus. This implementation is *experimental* and does
not yet achieve a reasonable trade-off between perplexity and differential 
privacy.

## Overview

This directory contains a routine to refactor the Huggingface/torch-transformer
implementation of GPT-2 into a module that is compatible with Opacus. This 
module also comes with built-in control & a utility that makes it easy to 
train the top k layers of GPT-2 (for a user selected k). Finally, we also 
include a low-rank factorization of the GPT-2 output layer (with the intent of 
reducing the # of parameters that DP-SGD fine-tunes).

## Instructions
Begin by moving into the `opacus-lab/GPT2/` directory.

### Requirements
Other than Pytorch and Opacus the only requirement is `transformers==4.7`.

### Python Path
Set `export PYTHONPATH=~/Opacus-lab/` or whatever path the Opacus-lab base
directory is in.

### Download Data
Run `bash prepare-wikitext-103.sh`

You can skip this if you already have the dataset downloaded.

### Preprocess Data
Run `python preprocess-wikitext-103.py`

This converts the raw text files into Pytorch tensors using on the GPT tokenizer.

### Train Model
Run `python run.py`

You can use the `-h` flag to learn more about the various arguments that are
currently supported.

## Block Level vs. User Level Privacy
For now, we only support block-level privacy. Generally speaking, for the
goal of fine-tuning on some user's private text, we'd like to consider
privacy at the user level rather than the block level. This is a stronger
notion of privacy, so achieving a reasonable trade-off between perplexity
and block-level privacy should be easier. In the near future, we aim to include
support for the `UsrLvlDataset` class in `GPT2/dataset.py`. 


## Code Acknowledgements 
We acknowledge the following code (all under the same Apache 2.0 License):
- affjljoo3581/GPT2 for the base GPT-2 Pytorch implementation that we modify
- pytorch/fairseq for the utilty script for downloading wikitext-103
- huggingface/transformers for the GPT-2 Pytorch implementation & pre-trained weights
that we refactor into
