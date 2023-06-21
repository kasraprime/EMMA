import os, json, csv
from re import M
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import tikzplotlib
import seaborn as sns
import pickle


# set the name of the experiment
name = 'converged-partial-train' # 'epochs' or 'converged-partial-train' or 'converged-batch-size' or 'converged-partial-train-loss-vs-mrr' 'converged-weighted-emma' 'converged-weighted-emma-scatter' 'converged-weighted-emma-threshold'
# name = 'converged-weighted-emma'
name = 'converged-weighted-emma-threshold'
name = 'converged-weighted-emma-threshold-scatter'
# name = 'converged-equally-weighted-emma'
# name = 'converged-equally-weighted-emma-optimizer-hyperparam'
name = 'converged-partial-train-loss-vs-mrr'
# name = 'converged-weighted-emma'

portion = 'test'

# Add and choose the experiments you want
exp_ugly_names = {
    'EMMAbert-0.7': 'exp-0.993-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAbertbs4-1': 'exp-0.99-train-supcon-emma-lard-4-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-1': 'exp-0.99-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-2': 'exp-0.98-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'EMMAbert-5': 'exp-0.95-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAt5base-5': 'exp-0.95-train-t5-base-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'EMMAbert-10': 'exp-0.90-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAt5base-10': 'exp-0.90-train-t5-base-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAbertW7-10': 'exp-0.90-train-weighted-0.7-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'EMMAbert-25': 'exp-0.75-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAt5base-25': 'exp-0.75-train-t5-base-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAbartbase-25': 'exp-0.75-train-bart-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAbartlarge-25': 'exp-0.75-train-bart-large-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'EMMAbert-30': 'exp-0.70-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-35': 'exp-0.65-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-45': 'exp-0.55-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-50': 'exp-0.5-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-75': 'exp-0.25-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'EMMAbert-100': 'exp-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAt5base-100': 'exp-t5-base-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAbartbase-100': 'exp-train-bart-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAbartlarge-100': 'exp-bart-large-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    # 'SupConbertD7-1': 'exp-0.99-train-dropout7-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbertD5-1': 'exp-0.99-train-dropout5-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbertZ-1': 'exp-0.99-train-zero-out-neg-row-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbertZS-1': 'exp-0.99-train-zero-out-neg-row-sym-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbertbs4-1': 'exp-0.99-train-supcon-lard-4-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbert4GradAccum-1': 'exp-0.99-train-grad-accum-supcon-lard-4-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupConbert-1': 'exp-0.99-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupConbert-2': 'exp-0.98-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'SupConbert-5': 'exp-0.95-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCont5base-5': 'exp-0.95-train-t5-base-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'SupConbert-10': 'exp-0.90-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCont5base-10': 'exp-0.90-train-t5-base-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'SupConbert-25': 'exp-0.75-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCont5base-25': 'exp-0.75-train-t5-base-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbartbase-25': 'exp-0.75-train-bart-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbartlarge-25': 'exp-0.75-train-bart-large-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    
    'SupConbert-30': 'exp-0.70-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupConbert-35': 'exp-0.65-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupConbert-45': 'exp-0.55-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupConbert-50': 'exp-0.5-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupConbert-75': 'exp-0.25-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    
    'SupConbert-100': 'exp-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCont5base-100': 'exp-t5-base-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbartbase-100': 'exp-train-bart-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupConbartlarge-100': 'exp-bart-large-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'Geometricbert-0.7': 'exp-0.993-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometricbert4-1': 'exp-0.99-train-full-emma-lard-4-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometricbert-1': 'exp-0.99-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometricbert-2': 'exp-0.98-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'Geometricbert-5': 'exp-0.95-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometrict5base-5': 'exp-0.95-train-t5-base-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'Geometricbert-10': 'exp-0.90-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometrict5base-10': 'exp-0.90-train-t5-base-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
 
    'Geometricbert-25': 'exp-0.75-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometrict5base-25': 'exp-0.75-train-t5-base-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometricbartlarge-25': 'exp-0.75-train-bart-large-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'Geometricbert-30': 'exp-0.70-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometricbert-35': 'exp-0.65-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometricbert-45': 'exp-0.55-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometricbert-50': 'exp-0.50-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometricbert-75': 'exp-0.25-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometricbert-100': 'exp-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    # 'Geometrict5base-100': 'exp-t5-base-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometrictbartlarge-100': 'exp-bart-large-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    # 'Contrastivebert-100': 'exp-contrastive-org-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
}



exp_ugly_names_batch_size = {
    'EMMA-4': 'exp-0.99-train-supcon-emma-lard-4-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMA-64': 'exp-0.99-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    'SupCon-4': 'exp-0.99-train-supcon-lard-4-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupConGradAccum-4': 'exp-0.99-train-grad-accum-supcon-lard-4-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupCon-64': 'exp-0.99-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    
    'Geometric-4': 'exp-0.99-train-full-emma-lard-4-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometric-64': 'exp-0.99-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
}

exp_ugly_names_weighted_emma_SGD = {
    'EMMAbert-0:1': 'exp-0.90-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.1:0.9': 'exp-0.90-train-weighted-0.1-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.2:0.8': 'exp-0.90-train-weighted-0.2-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.3:0.7': 'exp-0.90-train-weighted-0.3-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.5:0.5': 'exp-0.90-train-weighted-0.5-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAbert-1:1': 'exp-0.90-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.7:0.3': 'exp-0.90-train-weighted-0.7-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-1:0': 'exp-0.90-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
}
exp_ugly_names_weighted_emma = {
    'EMMAbert-0:1': 'exp-0.90-train-weighted-0.0-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.1:0.9': 'exp-0.90-train-weighted-0.1-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.2:0.8': 'exp-0.90-train-weighted-0.2-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.3:0.7': 'exp-0.90-train-weighted-0.3-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.5:0.5': 'exp-0.90-train-weighted-0.5-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMAbert-1:1': 'exp-0.90-train-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.7:0.3': 'exp-0.90-train-weighted-0.7-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-1:0': 'exp-0.90-train-weighted-1.0-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
}



exp_ugly_names_equally_weighted_emma = {
    'EMMAbert-0.1:0.1': 'exp-0.90-train-equally-weighted-0.1-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-0.5:0.5': 'exp-0.90-train-weighted-0.5-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-1:1': 'exp-0.90-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-10:10': 'exp-0.90-train-equally-weighted-10-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-100:100': 'exp-0.90-train-equally-weighted-100-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
}

exp_ugly_names_equally_weighted_emma_optimizer_hyperparam = {
    'EMMAbert-Adam_no_scheduler>0.01': 'exp-0.90-train-weighted-0.5-no-scheduler-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-Adam>0.001': 'exp-0.90-train-weighted-0.5-supcon-emma-lard-64-relu-Adam-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-Adam>0.01': 'exp-0.90-train-weighted-0.5-supcon-emma-lard-64-relu-Adam-0.01-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-Adam>0.1': 'exp-0.90-train-weighted-0.5-supcon-emma-lard-64-relu-Adam-0.1-unique_objects-gold-no_neg_sampling-1024',
    'EMMAbert-SGD>0.05': 'exp-0.90-train-weighted-0.5-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
}

# exp_ugly_names_weighted_emma = {
#     'EMMAbert-0.0': 'exp-0.90-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
#     # 'EMMAbert-0.1': 'exp-0.90-train-weighted-0.1-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
#     'EMMAbert-0.3': 'exp-0.90-train-weighted-0.3-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
#     'EMMAbert-0.5': 'exp-0.90-train-weighted-0.5-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
#     'EMMAbert-0.7': 'exp-0.90-train-weighted-0.7-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
#     'EMMAbert-1': 'exp-0.90-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
#     'EMMAbert_Equal_Weights-1': 'exp-0.90-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
# }

if name == 'converged-batch-size':
    exp_ugly_names = exp_ugly_names_batch_size
elif name  == 'converged-weighted-emma' or name == 'converged-weighted-emma-scatter' or name == 'converged-weighted-emma-threshold' or name == 'converged-weighted-emma-threshold-scatter':
    exp_ugly_names = exp_ugly_names_weighted_emma
elif name == 'converged-equally-weighted-emma':
    exp_ugly_names = exp_ugly_names_equally_weighted_emma
elif name == 'converged-equally-weighted-emma-optimizer-hyperparam':
    exp_ugly_names = exp_ugly_names_equally_weighted_emma_optimizer_hyperparam

# if you don't want to report results for different language models:
temp_exp_ugly_names = {}
for key in exp_ugly_names.keys():
    temp_exp_ugly_names[key.split('-')[0].replace('bert', '')+'-'+key.split('-')[1]] = exp_ugly_names[key]
    
exp_ugly_names = temp_exp_ugly_names

# exp_ugly_names = {
#     '4M-eMMA-text': '4M-eMMA-cosine-submodalities-text-anchor-gold-no_neg_sampling-1024',
#     '3M-eMMA-text': '3M-eMMA-cosine-submodalities-text-anchor-gold-no_neg_sampling-1024',
#     '4M-SupCon-text': '4M-supervised-contrastive-SGD-cosine-submodalities-text-first-gold-no_neg_sampling-1024',
#     '3M-SupCon-text': '3M-supervised-contrastive-SGD-cosine-submodalities-text-first-gold-no_neg_sampling-1024',
#     '4M-contrastive-text': '4M-contrastive-cosine-submodalities-text-anchor-gold-no_neg_sampling-1024',
#     '3M-contrastive-text': '3M-contrastive-SGD-cosine-submodalities-text-anchor-gold-no_neg_sampling-1024',
# }

# exp_ugly_names = {
    # '3M-triplet-text':  'triplet-cosine-Adam-text-anchor-gold-gold-no_neg_sampling-1024',
    # '3M-eMMA-audio': '3M-eMMA-cosine-submodalities-audio-anchor-gold-no_neg_sampling-1024',
    # '3m-SupCon-audio': '3M-supervised-contrastive-SGD-cosine-submodalities-audio-first-gold-no_neg_sampling-1024',
    # '4M-simple-MMA': '4M-mma-simple-SGD-cosine-submodalities-text-anchor-gold-no_neg_sampling-1024',
#     '3M-triplet-audio': 'triplet-cosine-Adam-audio16-anchor-gold-gold-no_neg_sampling-1024',
#     '3M-eMMA-audio': '3M-eMMA-cosine-submodalities-audio-anchor-gold-no_neg_sampling-1024',
#     '3M-eMMA-audio-Adam': '3M-eMMA-Adam-cosine-submodalities-audio-anchor-gold-no_neg_sampling-1024', # this should beat triplet.
#     '3m-SupCon-audio': '3M-supervised-contrastive-SGD-cosine-submodalities-audio-first-gold-no_neg_sampling-1024',
# }

experiments = {name:{} for name in exp_ugly_names.keys()}


for method in experiments.keys():
    for seed in [7, 24, 42, 123, 3407]:
        experiments[method]['seed-'+str(seed)] = glob(os.path.join('results', f'{exp_ugly_names[method]}/seed-{str(seed)}/'), recursive=True)[0]
        # experiments[method]['seed-'+str(seed)] = glob(os.path.join('/nfs/ada/ferraro/p/work/kasra/MMA/Code/results', f'{exp_ugly_names[method]}/seed-{str(seed)}/'), recursive=True)[0]
        # experiments[method]['seed-'+str(seed)] = glob(os.path.join('results', f'*{exp_ugly_names[method]}*/seed-{str(seed)}/'), recursive=True)[0]

print(experiments)

def get_color_style(method):
    # default values
    color = 'tab:blue'
    style = '--*'
    
    if 'SupCon' in method:
        color = 'tab:orange'
    elif 'EMMA' in method:
        color = 'tab:green'
    elif 'Geometric' in method:
        color = 'tab:blue'

    if 'bert' in method:
        style = '-.^'
    elif 't5base' in method:
        style = ':+'
    
    if 'bs4' in method:
        style = '-.o'
    if 'GradAccum' in method:
        style = '--v'
    if 'bertZ' in method:
        style = '--s'
    if 'bertZS' in method:
        style = '--8'
    if 'bertD5' in method:
        style = '-.H'
    if 'bertD7' in method:
        style = '--*'
    if 'W7' in method:
        style = '-.o'
    if 'Equal' in method:
        style = '-.o'
    
    return color, style

results = {}
logs = {}
outputs = {}
for method in experiments.keys():
    results[method] = {}
    logs[method] = {}
    outputs[method] = {}
    for seed in experiments[method].keys():
        res = json.load(open(experiments[method][seed]+'results.json'))
        loss = json.load(open(experiments[method][seed]+'logs.json'))
        if name == 'converged-weighted-emma-threshold' or name == 'converged-weighted-emma-threshold-scatter':
            f = open(experiments[method][seed]+'outputs-test.pkl', 'rb')
            output = pickle.load(f)
            outputs[method][seed] = output[portion]['thresh_percent']
            f.close()
        del res['best']
        if 'test-only' in res.keys():
            del res['test-only']
        results[method][seed] = res
        logs[method][seed] = {key: value['total'] for key, value in loss.items()}

    results[method]['avg'] = {'best':{}}
    results[method]['std'] = {'best':{}}
    logs[method]['avg'] = {'best':{}}
    logs[method]['std'] = {'best':{}}
    # computing average and std over seeds for each metric of each portion of each epoch
    for epoch in results[method][seed].keys():
        results[method]['avg'][epoch] = {}
        results[method]['std'][epoch] = {}
        logs[method]['avg'][epoch] = np.mean([logs[method][seed][epoch] for seed in experiments[method].keys()])
        logs[method]['std'][epoch] = np.std([logs[method][seed][epoch] for seed in experiments[method].keys()])
        for portion in results[method]['seed-42'][epoch].keys():
            results[method]['avg'][epoch][portion] = {}
            results[method]['std'][epoch][portion] = {}
            for metric in results[method]['seed-42'][epoch][portion].keys():
                results[method]['avg'][epoch][portion][metric] = np.mean([results[method][seed][epoch][portion][metric] for seed in experiments[method].keys()])
                results[method]['std'][epoch][portion][metric] = np.std([results[method][seed][epoch][portion][metric] for seed in experiments[method].keys()])
    
    # computing best for each metric and portion. We find the best average among all epochs but for the std we have to use the std corresponding to that best average.
    for portion in results[method]['seed-42']['0'].keys():
        results[method]['avg']['best'][portion] = {}
        results[method]['std']['best'][portion] = {}
        logs[method]['avg']['best'][portion] = {}
        logs[method]['std']['best'][portion] = {}

        for metric in results[method]['seed-42']['0'][portion].keys():
            results[method]['avg']['best'][portion][metric] = np.max([results[method]['avg'][epoch][portion][metric] for epoch in results[method][seed].keys()])
            best_idx = np.argmax([results[method]['avg'][epoch][portion][metric] for epoch in results[method][seed].keys()])
            results[method]['std']['best'][portion][metric] = [results[method]['std'][epoch][portion][metric] for epoch in results[method][seed].keys()][best_idx]
            
            # I could discard portion and metric since loss is for train and isn't related to the mrr, but since the best performance is chosen for each metric and poriton, the corresponding index should be used
            logs[method]['avg']['best'][portion][metric] = [logs[method]['avg'][epoch] for epoch in logs[method][seed].keys()][best_idx]
            logs[method]['std']['best'][portion][metric] = [logs[method]['std'][epoch] for epoch in logs[method][seed].keys()][best_idx]
    
    # metr = 'mrr_ard'
    # for metr in ['mrr', 'acc']:
    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard', 'acc_ad', 'acc_ar', 'acc_ld', 'acc_lr', 'acc_lad', 'acc_lar', 'acc_ard', 'acc_lrd', 'acc_lard']:
        print(f'method: {method}, best {metr} ** avg:', results[method]['avg']['best']['test'][metr], f'std:', results[method]['std']['best']['test'][metr])

print(f"----**** Done with loading results, logs, and ouptuts! ****----")
print(f" outputs:{outputs} ")
# Save results as csv and then convert it easily to latex table with best performances bolded.
path = 'result-analysis/'
metrics = ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard', 'acc_ad', 'acc_ar', 'acc_ld', 'acc_lr', 'acc_lad', 'acc_lar', 'acc_ard', 'acc_lrd', 'acc_lard']
# col = ['Method'] + [metr.replace('_','.') for metr in metrics] # underline _ mess with latex.
metrics_mrr = ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']
metrics_acc = ['acc_ad', 'acc_ar', 'acc_ld', 'acc_lr', 'acc_lad', 'acc_lar', 'acc_ard', 'acc_lrd', 'acc_lard']
col = ['Method'] + metrics_mrr 
result_mrr = pd.DataFrame(columns=col)
col = ['Method'] + metrics_acc
result_acc = pd.DataFrame(columns=col) 
best_bold = {}
for metr in metrics:
    idx = np.argmax([round(results[method]['avg']['best']['test'][metr]*100, 2) for method in results.keys()])
    best_bold[metr] = list(results.keys())[idx]

for idx, method in enumerate(list(results.keys())):
    result_mrr.at[idx, 'Method'] = method
    result_acc.at[idx, 'Method'] = method
    for metr in metrics_mrr:
        if method == best_bold[metr]:
            result_mrr.at[idx, metr] = f"\\textbf{{{round(results[method]['avg']['best']['test'][metr]*100, 2)}}}$\pm${round(results[method]['std']['best']['test'][metr]*100, 2)}"
        else:
            result_mrr.at[idx, metr] = f"{round(results[method]['avg']['best']['test'][metr]*100, 2)}$\pm${round(results[method]['std']['best']['test'][metr]*100, 2)}"
    for metr in metrics_acc:
        if method == best_bold[metr]:
            result_acc.at[idx, metr] = f"\\textbf{{{round(results[method]['avg']['best']['test'][metr]*100, 2)}}}$\pm${round(results[method]['std']['best']['test'][metr]*100, 2)}"
        else:
            result_acc.at[idx, metr] = f"{round(results[method]['avg']['best']['test'][metr]*100, 2)}$\pm${round(results[method]['std']['best']['test'][metr]*100, 2)}"

# underline _ mess with latex, (l)anguage --> (t)ext, and (a)udio --> (s)peech
columns_spell_out = {
    'lrd': 'text/RGB/\\\Depth',
    'ard': 'speech/RGB/\\\depth',
    'lar': 'text/speech/\\\RGB',
    'lad': 'text/speech/\\\depth',
    'lard': 'all',
    'lr': 'text/RGB',
    'ld': 'text/depth',
    'ar': 'speech/RGB',
    'ad': 'speech/depth',
}
# result_mrr.columns = ['Method'] + [f"{metr.split('_')[0].upper()} {metr.split('_')[1].replace('l', 't').replace('a', 's')}" for metr in metrics_mrr]
# result_acc.columns = ['Method'] + [f"{metr.split('_')[0].capitalize()} {metr.split('_')[1].replace('l', 't').replace('a', 's')}" for metr in metrics_acc]
result_mrr.columns = ['Method'] + [columns_spell_out[metr.split('_')[1]] for metr in metrics_mrr]
result_acc.columns = ['Method'] + [columns_spell_out[metr.split('_')[1]] for metr in metrics_acc]
result_mrr.to_csv(path_or_buf=path+'resultsMRR.csv', index=False)
result_acc.to_csv(path_or_buf=path+'resultsACC.csv', index=False)


markers = ['*', '<', 'o', 'P', '^', 's', '+', 'v', '>','8', 'p', 'h','H','D','d','X']
styles = ['-',':','--','-.','|']
styles_markers = ['-.^', ':+', '-o', '--s', '-.o', '-+', '--+', '-.+', ':o', '-v', '--v', '-.v']
colors = ['tab:orange', 'tab:green', 'tab:blue', 'tab:red']
colors = list(mcd.XKCD_COLORS.values())[-30:]
colors.reverse()
# colors = list(mcd.TABLEAU_COLORS.keys())
# colors = {'f1': 'magenta', 'precision': 'red', 'recall':'blue'}



metric_correct_names = {
    'mrr_lrd': 'MRR speech ablated (trd)',
    'mrr_ard': 'MRR text ablated (srd)',
    'mrr_lar': 'MRR depth ablated (tsr)',
    'mrr_lad': 'MRR RGB ablated (tsd)',
    'mrr_lard': 'MRR all modalities (tsrd)',
    'mrr_lr': 'MRR speech and depth ablated (tr)',
    'mrr_ld': 'MRR speech and RGB ablated (td)',
    'mrr_ar': 'MRR text and depth ablated (sr)',
    'mrr_ad': 'MRR text and RGB ablated  (sd)',
}

# Exponentially sampled epochs to show more points in the first half
# epochs = list(results['EMMA']['avg'].keys())
# epochs.remove('best')
# exp_sampled = np.random.exponential(scale=200, size=len(epochs))
# unit_exp_prob = (exp_sampled + min(exp_sampled)) / sum(exp_sampled + min(exp_sampled))
# epochs = list(np.random.choice(epochs, size=50, replace=False, p=unit_exp_prob))
# epochs.sort(key = int)
# print(epochs)

if name == 'epochs':
    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
    # for metr in ['mrr_ard']:
        fig = plt.figure(figsize=(25,20))
        ax = fig.add_subplot(1,1,1)
        for method_idx, method in enumerate(results.keys()):
            epochs = list(results[method]['avg'].keys())
            plt.plot([str(epoch) for epoch in epochs if epoch != 'best'], [results[method]['avg'][epoch][portion][metr] for epoch in epochs if epoch != 'best'], styles_markers[method_idx], label=method, markersize=15)
            low = np.asarray([results[method]['avg'][epoch][portion][metr] for epoch in epochs if epoch != 'best']) - np.asarray([results[method]['std'][epoch][portion][metr] for epoch in epochs if epoch != 'best'])
            high = np.asarray([results[method]['avg'][epoch][portion][metr] for epoch in epochs if epoch != 'best']) + np.asarray([results[method]['std'][epoch][portion][metr] for epoch in epochs if epoch != 'best'])
            plt.fill_between([str(epoch) for epoch in epochs if epoch != 'best'], low, high, alpha=0.2)

        # plt.title('Metrics for different threshold of label inclusion excluding object w/ stop, w/ attention, w/o VAE, and using word embeddings')
        # ax.grid(True)
        # ax.set_xlabel('Epoch', fontsize=40)
        # ax.set_ylabel(metr, fontsize=40)
        # plt.xticks(np.arange(0, len(list(results[method]['avg'].keys()))+1, 10), fontsize=30, rotation=90)
        # plt.yticks(fontsize=30)
        plt.grid(True)
        plt.xlabel('Epoch', fontsize=40)
        plt.ylabel(metric_correct_names[metr], fontsize=40)
        plt.xscale("symlog", base=2)
        plt.xticks(fontsize=30)
        # plt.xticks([1, 5, 10, 50, 80, 100, 150, 190], fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(loc='lower right', fontsize=40, ncol=1)
        # fig.legend(loc='lower right', fontsize=40, ncol=1)
        plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')

elif name == 'converged-partial-train':
    results_converged = {}
    for method in results.keys():
        if method.split('-')[0] not in results_converged.keys():
            results_converged[method.split('-')[0]] = {}
        
        if method.split('-')[1] not in results_converged[method.split('-')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1]] = results[method]

    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
        fig = plt.figure(figsize=(25,20))
        ax = fig.add_subplot(1,1,1)
        for method_idx, method in enumerate(results_converged.keys()):
            percentages = list(results_converged[method].keys())
            # plt.plot([str(percent) for percent in percentages], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in percentages], styles_markers[method_idx], label=method, markersize=15)
            color_model, style_embd = get_color_style(method)
            plt.plot([str(int(float(percent)*73.80)) for percent in percentages], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in percentages], style_embd, color=color_model, label=method, markersize=15)
            low = np.asarray([results_converged[method][percent]['avg']['best'][portion][metr] for percent in percentages]) - np.asarray([results_converged[method][percent]['std']['best'][portion][metr] for percent in percentages])
            high = np.asarray([results_converged[method][percent]['avg']['best'][portion][metr] for percent in percentages]) + np.asarray([results_converged[method][percent]['std']['best'][portion][metr] for percent in percentages])
            plt.fill_between([str(int(float(percent)*73.80)) for percent in percentages], low, high, color=color_model, alpha=0.2)

        # plt.title('Metrics for different threshold of label inclusion excluding object w/ stop, w/ attention, w/o VAE, and using word embeddings')
        # ax.grid(True)
        # ax.set_xlabel('Epoch', fontsize=40)
        # ax.set_ylabel(metr, fontsize=40)
        # plt.xticks(np.arange(0, len(list(results[method]['avg'].keys()))+1, 10), fontsize=30, rotation=90)
        # plt.yticks(fontsize=30)
        plt.grid(True)
        plt.xlabel('Number of Training Data Points Used', fontsize=40)
        plt.ylabel(metric_correct_names[metr], fontsize=40)
        # plt.xscale("symlog", base=2)
        plt.xticks(fontsize=30)
        # plt.xticks([1, 5, 10, 50, 80, 100, 150, 190], fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(loc='lower right', fontsize=30, ncol=1)
        # fig.legend(loc='lower right', fontsize=40, ncol=1)
        plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')


elif name == 'converged-batch-size':
    results_converged = {}
    for method in results.keys():
        if method.split('-')[0] not in results_converged.keys():
            results_converged[method.split('-')[0]] = {}
        
        if method.split('-')[1] not in results_converged[method.split('-')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1]] = results[method]

    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
        fig = plt.figure(figsize=(25,20))
        ax = fig.add_subplot(1,1,1)
        for method_idx, method in enumerate(results_converged.keys()):
            batch_size = list(results_converged[method].keys())
            # plt.plot([str(percent) for percent in batch_size], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in batch_size], styles_markers[method_idx], label=method, markersize=15)
            color_model, style_embd = get_color_style(method)
            plt.plot([str(percent) for percent in batch_size], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in batch_size], style_embd, color=color_model, label=method, markersize=15)
            low = np.asarray([results_converged[method][percent]['avg']['best'][portion][metr] for percent in batch_size]) - np.asarray([results_converged[method][percent]['std']['best'][portion][metr] for percent in batch_size])
            high = np.asarray([results_converged[method][percent]['avg']['best'][portion][metr] for percent in batch_size]) + np.asarray([results_converged[method][percent]['std']['best'][portion][metr] for percent in batch_size])
            plt.fill_between([str(percent) for percent in batch_size], low, high, color=color_model, alpha=0.2)

        plt.grid(True)
        plt.xlabel('Batch Size', fontsize=40)
        plt.ylabel(metric_correct_names[metr], fontsize=40)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(loc='upper right', fontsize=30, ncol=1)
        plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')



elif name == 'converged-partial-train-loss-vs-mrr':
    results_converged = {}
    logs_converged = {}
    for method in results.keys():
        if method.split('-')[0] not in results_converged.keys():
            results_converged[method.split('-')[0]] = {}
        if method.split('-')[0] not in logs_converged.keys():
            logs_converged[method.split('-')[0]] = {}
        
        if method.split('-')[1] not in results_converged[method.split('-')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1]] = results[method]
        if method.split('-')[1] not in logs_converged[method.split('-')[0]].keys():
            logs_converged[method.split('-')[0]][method.split('-')[1]] = logs[method]

    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
        fig = plt.figure(figsize=(25,20))
        ax = fig.add_subplot(1,1,1)
        df = pd.DataFrame(columns=['Method', 'Data', 'Loss', 'MRR'])        
        
        for method_idx, method in enumerate(results_converged.keys()):
            percentages = list(results_converged[method].keys())
        #     methods = list(np.repeat(list(results_converged.keys()), len(percentages)))
            # plt.plot([str(percent) for percent in percentages], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in percentages], styles_markers[method_idx], label=method, markersize=15)
            # color_model, _ = get_color_style(method)
            # #plt.scatter([logs_converged[method][percent]['avg']['best'][portion][metr] for percent in percentages], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in percentages], markers, color=color_model, label=method, markersize=15)
            
            for percent_idx, percent in enumerate(percentages):
            #     # plt.scatter(logs_converged[method][percent]['avg']['best'][portion][metr], results_converged[method][percent]['avg']['best'][portion][metr], marker=markers[percent_idx], color=color_model, label=method, s=70)
            #     plt.scatter(logs_converged[method][percent]['avg']['best'][portion][metr], results_converged[method][percent]['avg']['best'][portion][metr], marker=markers[method_idx], color=colors[percent_idx], label=method, s=70)

                temp = {'Method': [method], 'Data': [int(float(percent)*73.80)], 'Loss': [logs_converged[method][percent]['avg']['best'][portion][metr]], 'MRR': [results_converged[method][percent]['avg']['best'][portion][metr]]}
                df = pd.concat([df, pd.DataFrame(temp)], ignore_index=True)
        
        # sns.scatterplot(x=[logs_converged[method][percent]['avg']['best'][portion][metr] for percent in percentages], y=[results_converged[method][percent]['avg']['best'][portion][metr]for percent in percentages], hue=[str(int(float(percent)*73.80)) for percent in percentages], style=methods, palette='crest', s=70)
        sns.scatterplot(data=df, x='Loss', y='MRR', hue='Data', style='Method', palette='crest', s=90)
        # plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
        # plt.setp(ax.get_legend().get_title(), fontsize='30') # for legend title
        plt.grid(True)
        plt.xlabel('Converged Loss', fontsize=40)
        plt.ylabel(metric_correct_names[metr], fontsize=40)
        # plt.xscale("symlog", base=2)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # plt.legend(loc='upper right', fontsize=25, ncol=2)


        ax = plt.gca()
        ax.legend_.remove()
        # Create a new legend for style only
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[labels.index('Method') : len(labels)], labels[labels.index('Method') : len(labels)], fontsize=25)
        
        sm = plt.cm.ScalarMappable(cmap='crest', norm=plt.Normalize(vmin=min(df['Data']), vmax=max(df['Data'])))
        sm.set_array([])  # Set an empty array to define the range
        ## Add the color bar legend
        cbar = plt.colorbar(sm)
        ## Set the label for the color bar
        cbar.set_label("Number of Training Data", fontsize=25)
        ## Set the ticks and labels for the color bar
        # cbar.set_ticks([min(weights['weights']), max(weights['weights'])])
        # cbar.set_ticklabels([str(min(weights['weights'])), str(max(weights['weights']))])
        cbar.set_ticks(list(np.unique(df['Data']))[1::2])
        cbar.set_ticklabels([str(w) for w in list(np.unique(df['Data']))[1::2]])
        cbar.ax.tick_params(labelsize=25)
        # plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')

        # {   colors methods, markers percents
        # marker_percents = list(list(results_converged.values())[0].keys())
        # handles_percents = [plt.plot([], [], markers[i], markerfacecolor='w', markeredgecolor='k')[0] for i in range(len(marker_percents))]
        # color_methods = list(results_converged.keys())
        # colors = []
        # for m in color_methods:
        #     color, _ = get_color_style(m)
        #     colors.append(color)
        # handles_methods = [plt.plot([], [], colors[i])[0] for i in range(len(colors))]
        # plt.legend(handles_percents+handles_methods, marker_percents+color_methods, loc='upper right', framealpha=1, fontsize=30, ncol=2)
        # }

        
        # marker_methods = list(results_converged.keys())
        # handles_methods = [plt.plot([], [], markers[i], markerfacecolor='w', markeredgecolor='k')[0] for i in range(len(marker_methods))]
        # color_percents = [str(int(float(percent)*73.80)) for percent in list(list(results_converged.values())[0].keys())]
        # # color_percents = list(list(results_converged.values())[0].keys())
        # handles_percents = [plt.plot([], [], colors[i])[0] for i in range(len(color_percents))]
        # plt.legend(handles_percents+handles_methods, color_percents+marker_methods, loc='upper right', framealpha=1, fontsize=30, ncol=2)

        plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')

elif name == 'converged-weighted-emma' or name == 'converged-equally-weighted-emma':
    results_converged = {}
    for method in results.keys():
        if method.split('-')[0] not in results_converged.keys():
            results_converged[method.split('-')[0]] = {}
        
        if method.split('-')[1] not in results_converged[method.split('-')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1]] = results[method]

    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
        fig = plt.figure(figsize=(25,20))
        ax = fig.add_subplot(1,1,1)
        for method_idx, method in enumerate(results_converged.keys()):
            weight = list(results_converged[method].keys())
            # plt.plot([str(percent) for percent in weight], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in weight], styles_markers[method_idx], label=method, markersize=15)
            color_model, style_embd = get_color_style(method)
            plt.plot([str(percent) for percent in weight], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in weight], style_embd, color=color_model, label=method, markersize=15)
            low = np.asarray([results_converged[method][percent]['avg']['best'][portion][metr] for percent in weight]) - np.asarray([results_converged[method][percent]['std']['best'][portion][metr] for percent in weight])
            high = np.asarray([results_converged[method][percent]['avg']['best'][portion][metr] for percent in weight]) + np.asarray([results_converged[method][percent]['std']['best'][portion][metr] for percent in weight])
            plt.fill_between([str(percent) for percent in weight], low, high, color=color_model, alpha=0.2)

        plt.grid(True)
        plt.xlabel('Impact Ratio of Geometric:SupCon', fontsize=40)
        # plt.xlabel('Weight of Geometric Method in EMMA', fontsize=40)
        plt.ylabel(metric_correct_names[metr], fontsize=40)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(loc='lower right', fontsize=30, ncol=1)
        plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')



elif name == 'converged-weighted-emma-scatter':
    results_converged = {}
    for method in results.keys():
        if method.split('-')[0] not in results_converged.keys():
            results_converged[method.split('-')[0]] = {}
        
        if method.split('-')[1] not in results_converged[method.split('-')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1]] = results[method]

    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
        # fig = plt.figure(figsize=(15,20))
        fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        for method_idx, method in enumerate(results_converged.keys()):
            weight = list(results_converged[method].keys())
            # plt.plot([str(percent) for percent in weight], [results_converged[method][percent]['avg']['best'][portion][metr] for percent in weight], styles_markers[method_idx], label=method, markersize=15)
            # color_model, style_embd = get_color_style(method)
            sns.scatterplot(x=[float(percent.split(':')[0]) for percent in weight], y=[float(percent.split(':')[1]) for percent in weight], hue=[round(results_converged[method][percent]['avg']['best'][portion][metr],2) for percent in weight], size=[round(results_converged[method][percent]['avg']['best'][portion][metr],2) for percent in weight], palette="deep")

        plt.grid(True)
        # plt.xlabel('Weight of Geometric Method in EMMA', fontsize=40)
        plt.xlabel('Weight of Geometric Method in EMMA')
        # plt.ylabel('Weight of SupCon Method in EMMA', fontsize=40)
        plt.ylabel('Weight of SupCon Method in EMMA')
        # plt.title(metric_correct_names[metr], fontsize=40)
        plt.title(metric_correct_names[metr])
        # plt.xticks(fontsize=30)
        # plt.xticks([float(percent.split(':')[0]) for percent in weight], fontsize=30)
        plt.xticks([float(percent.split(':')[0]) for percent in weight])
        # plt.yticks(fontsize=30)
        # plt.yticks([float(percent.split(':')[1]) for percent in weight], fontsize=30)
        plt.yticks([float(percent.split(':')[1]) for percent in weight])
        # plt.legend(loc='lower left', fontsize=30, ncol=1)
        plt.legend(loc='lower left')
        plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')

elif name == 'converged-weighted-emma-threshold':
    results_converged = {}
    for method in outputs.keys():
        if method.split('-')[0] not in results_converged.keys():
            results_converged[method.split('-')[0]] = {}
        
        if method.split('-')[1] not in results_converged[method.split('-')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1]] = outputs[method]

    print(f" results_converged:{results_converged} ")
    fig = plt.figure(figsize=(25,20))
    ax = fig.add_subplot(1,1,1)
    for method_idx, method in enumerate(results_converged.keys()):
        print(f" method:{method} ")
        weight = list(results_converged[method].keys())
        color_model, style_embd = get_color_style(method)
        plt.plot([str(percent) for percent in weight], [np.mean([results_converged[method][percent][seed] for seed in results_converged[method][percent].keys()]) for percent in weight], style_embd, color=color_model, label=method, markersize=15)
        low = np.asarray([np.mean([results_converged[method][percent][seed] for seed in results_converged[method][percent].keys()]) for percent in weight]) - np.asarray([np.std([results_converged[method][percent][seed] for seed in results_converged[method][percent].keys()]) for percent in weight])
        high = np.asarray([np.mean([results_converged[method][percent][seed] for seed in results_converged[method][percent].keys()]) for percent in weight]) + np.asarray([np.std([results_converged[method][percent][seed] for seed in results_converged[method][percent].keys()]) for percent in weight])
        plt.fill_between([str(percent) for percent in weight], low, high, color=color_model, alpha=0.2)

    plt.grid(True)
    plt.xlabel('Impact Ratio of Geometric:SupCon', fontsize=40)
    # plt.xlabel('Weight of Geometric Method in EMMA', fontsize=40)
    plt.ylabel('Percentage Thresholded', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(loc='lower right', fontsize=30, ncol=1)
    plt.savefig('result-analysis/average-seeds-'+name+'.pdf')



if name == 'converged-weighted-emma-threshold-scatter':
    results_converged = {}
    outputs_converged = {}
    for method in results.keys():
        if method.split('-')[0] not in results_converged.keys():
            results_converged[method.split('-')[0]] = {}
            outputs_converged[method.split('-')[0]] = {}
        
        if method.split('-')[1] not in results_converged[method.split('-')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1]] = results[method]
            outputs_converged[method.split('-')[0]][method.split('-')[1]] = outputs[method]

    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
        # fig = plt.figure(figsize=(15,20))
        fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        
        for method_idx, method in enumerate(results_converged.keys()):
            weight = list(results_converged[method].keys())
            
            weights = {'weights': [float(percent.split(':')[0]) for percent in weight]}
            
            sns.scatterplot(x=[np.mean([outputs_converged[method][geom_weight][seed] for seed in outputs_converged[method][geom_weight].keys()]) for geom_weight in weight], y=[results_converged[method][geom_weight]['avg']['best'][portion][metr] for geom_weight in weight], hue=weights['weights'], palette='crest')
            # sns.scatterplot(x=[np.mean([outputs_converged[method][geom_weight][seed] for seed in outputs_converged[method][geom_weight].keys()]) for geom_weight in weight], y=[results_converged[method][geom_weight]['avg']['best'][portion][metr] for geom_weight in weight], hue=weights['weights'], style=weights['weights'], palette="deep")
            # for weight_idx, geom_weight in enumerate(weight):
                # plt.scatter(np.mean([outputs_converged[method][geom_weight][seed] for seed in outputs_converged[method][geom_weight].keys()]), results_converged[method][geom_weight]['avg']['best'][portion][metr], marker='o', color=colors[weight_idx], label=weights['weights'][weight_idx], s=70)
        plt.grid(True)
        plt.xlabel('Percentage of Dissimilar Items With a Certain Distance Between Them')
        plt.ylabel(metric_correct_names[metr])
        # plt.ylabel(metric_correct_names[metr], fontsize=40)
        # plt.title('Percentage vs. MRR')
        
        # plt.xticks([float(percent.split(':')[0]) for percent in weight])
        plt.xticks(np.arange(0, 85, 10))
        plt.yticks(np.arange(0.50, 0.95, 0.05))
        # plt.legend(loc='lower left', fontsize=30, ncol=1)
        # plt.legend(loc='lower right', title='Weight of Geom')
        

        # color bar legend
        ## Get the current axes of the plot
        ax = plt.gca()
        ## Remove the default legend
        ax.legend_.remove()
        ## Create a color bar legend
        sm = plt.cm.ScalarMappable(cmap='crest', norm=plt.Normalize(vmin=min(weights['weights']), vmax=max(weights['weights'])))
        sm.set_array([])  # Set an empty array to define the range
        ## Add the color bar legend
        cbar = plt.colorbar(sm)
        ## Set the label for the color bar
        cbar.set_label("Weight of Geom")
        ## Set the ticks and labels for the color bar
        # cbar.set_ticks([min(weights['weights']), max(weights['weights'])])
        # cbar.set_ticklabels([str(min(weights['weights'])), str(max(weights['weights']))])
        cbar.set_ticks(weights['weights'])
        cbar.set_ticklabels([str(w) for w in weights['weights']])


        plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')


elif name == 'converged-equally-weighted-emma-optimizer-hyperparam':
    results_converged = {}
    for method in results.keys():
        if method.split('-')[0] not in results_converged.keys():
            results_converged[method.split('-')[0]] = {}
        
        if method.split('-')[1].split('>')[0] not in results_converged[method.split('-')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1].split('>')[0]] = {}
        
        if method.split('-')[1].split('>')[1] not in results_converged[method.split('-')[0]][method.split('-')[1].split('>')[0]].keys():
            results_converged[method.split('-')[0]][method.split('-')[1].split('>')[0]][method.split('-')[1].split('>')[1]] = results[method]
    print(f" results_converged.keys: {results_converged.keys()} \n optimiziers: {results_converged['EMMA'].keys()} ")
    for optim in list(results_converged['EMMA'].keys()):
        print(f"optim: {optim}, lrs: {results_converged['EMMA'][optim].keys()}")
        
    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
        fig = plt.figure(figsize=(25,20))
        ax = fig.add_subplot(1,1,1)
        for method_idx, method in enumerate(results_converged.keys()):
            # if more than one method, use colors for methods and marker for optimizers
            for optim_idx, optim in enumerate(results_converged[method].keys()):
                plt.plot([float(lr) for lr in list(results_converged[method][optim].keys())], [results_converged[method][optim][lr]['avg']['best'][portion][metr] for lr in list(results_converged[method][optim].keys())], marker=markers[optim_idx], color=colors[optim_idx], label=optim, markersize=15)
                low = np.asarray([results_converged[method][optim][lr]['avg']['best'][portion][metr] for lr in list(results_converged[method][optim].keys())]) - np.asarray([results_converged[method][optim][lr]['std']['best'][portion][metr] for lr in list(results_converged[method][optim].keys())])
                high = np.asarray([results_converged[method][optim][lr]['avg']['best'][portion][metr] for lr in list(results_converged[method][optim].keys())]) + np.asarray([results_converged[method][optim][lr]['std']['best'][portion][metr] for lr in list(results_converged[method][optim].keys())])
                plt.fill_between([float(lr) for lr in list(results_converged[method][optim].keys())], low, high, color=colors[optim_idx], alpha=0.2)

        plt.grid(True)
        plt.xlabel('Learning Rate', fontsize=40)
        # plt.xlabel('Weight of Geometric Method in EMMA', fontsize=40)
        plt.ylabel(metric_correct_names[metr], fontsize=40)
        plt.xticks([0.001, 0.01, 0.05, 0.1], fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(loc='lower right', fontsize=30, ncol=1)
        plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')

# Frank's request
for method_idx, method in enumerate(results.keys()):
    for seed in experiments[method].keys():
        del results[method][seed]
    for key in results[method].keys():
        del results[method][key]['best']
        for epoch in results[method][key].keys(): 
            del results[method][key][epoch]['valid']
json.dump(results, open('result-analysis/fig3-results.json', 'w'), indent=4)
df = pd.read_json('result-analysis/fig3-results.json')
df.to_csv('result-analysis/fig3-results.csv')
