# Optimization of Multi-Agent Flying Sidekick Traveling Salesman Problem over Road Networks

[**Ruixiao Yang**](https://scholar.google.com/citations?user=c0W8nfwAAAAJ), [**Chuchu Fan**](https://aeroastro.mit.edu/people/chuchu-fan/)

This repository provides the official implementation of our paper, "Optimization of Multi-Agent Flying Sidekick Traveling Salesman Problem over Road Networks
"[[PDF](https://arxiv.org/pdf/2408.11187)]

## Installation
Clone the repository:
```bash
git clone https://github.com/Brelliothe/MixTSP.git
```
Run the following command to install the required packages:
```bash
conda create -n MixTSP python=3.7
conda activate MixTSP
pip install -r requirements.txt
```

## Reproduce Guide
You can find our algorithm implemented in file `src/fstsp.py` and all other baselines in the same folder. 
To run all the experiments in the paper, you can use 
```bash
python experiments.py
```
To generate all the figures in the paper, you can use 
```bash
python plot.py
```

## Citation
If you find our research helpful for your work, please consider starring this repo and citing our paper.

