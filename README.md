# EMMA
This work is published at TMLR. [[```Paper```](https://openreview.net/forum?id=cXa6Xdm0v7)] [[```Citation```](#cite)]

EMMA stands for Extended Multimodal Alignment which is the idea to map the same concepts from different input data sources to the same shared latent space.

This repository contains three parts.
1. Code: where *the code* is located obviously, along with the results.
2. Data: where the raw text data and embeddings of all modalities (RGB, depth, text, and speech) are saved.
3. Paper: The LaTeX source code for the paper.


# How to run the Code
Install requierments.
```bash
pip install -r requirements.txt
```
Go to the *Code* directory first.
```bash
cd Code
```
Make a directory called *results*:
```bash
mkdir results
```
You have two options to run the code.
1. Run the code locally
2. Run it on a cluster using slurm

## Run on your own (super) computer
```bash
python -u ML.py --arg1_name $arg1_value --arg2_name $arg2_value ...
```
All arguments have a default value which you can see in the ML.py file, so you can leave arguemnts blank when running the ML.py file, or you can specify any of them you want.

## Run on a cluster
Make a directoy inside the ```Code/results``` directory and name it *jobs*. you only need to do this step once, and not everytime you run a job on cluster.
```bash
mkdir -p results/jobs
```
Use the following bash command. if you want to run only ONE job on the cluster.
```bash
bash run_cluster.sh
```
If you want to run more than one job run the following command. This can be used when you want to run your code with multiple random seeds or for different values for other hyperparameters.
```bash
bash run_cluster_loop.sh
```


## Dataset directory
The most simple way would be to give your dataset directory to the ```--data_dir``` argument when running ML.py.
The current setup assumes that the dataset (GoLD) is located in the following directory with respect to the Code directory:
```../../../../data/gold/```

## Arguments
### Methods
You can run 4 different methods by specifying their names in the arguments.
1. supcon-emma : Our method which we call EMMA and is essentially a combination of SupCon and our Geometric methods.
2. full-emma: Our proposed method which uses Geometric alignment to learn concepts.
3. supcon: State-of-the-art baseline
4. contrastive-org: Another SOTA baseline, but since it's not as strong as SupCon, we only report the results compared to SupCon.


# Cite
Please use the following bibtex entry to cite this work.

```
@article{
darvish2023multimodal,
title={Multimodal Language Learning for Object Retrieval in Low Data Regimes in the Face of Missing Modalities},
author={Kasra Darvish and Edward Raff and Francis Ferraro and Cynthia Matuszek},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=cXa6Xdm0v7},
note={}
}
```
