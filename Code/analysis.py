import os, json, csv
from re import M
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import tikzplotlib


exp_ugly_names = {
    'Geometric': 'exp-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupCon': 'exp-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMA': 'exp-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Contrastive': 'exp-contrastive-org-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
}


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
        # experiments[method]['seed-'+str(seed)] = glob(os.path.join('results', f'*{exp_ugly_names[method]}*/seed-{str(seed)}/'), recursive=True)[0]

print(experiments)


results = {}
for method in experiments.keys():
    results[method] = {}
    for seed in experiments[method].keys():
        res = json.load(open(experiments[method][seed]+'results.json'))
        del res['best']
        results[method][seed] = res
    results[method]['avg'] = {'best':{}}
    results[method]['std'] = {'best':{}}
    # computing average and std over seeds for each metric of each portion of each epoch
    for epoch in results[method][seed].keys():
        results[method]['avg'][epoch] = {}
        results[method]['std'][epoch] = {}
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
        for metric in results[method]['seed-42']['0'][portion].keys():
            results[method]['avg']['best'][portion][metric] = np.max([results[method]['avg'][epoch][portion][metric] for epoch in results[method][seed].keys()])
            best_idx = np.argmax([results[method]['avg'][epoch][portion][metric] for epoch in results[method][seed].keys()])
            results[method]['std']['best'][portion][metric] = [results[method]['std'][epoch][portion][metric] for epoch in results[method][seed].keys()][best_idx]
    
    # metr = 'mrr_ard'
    # for metr in ['mrr', 'acc']:
    for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard', 'acc_ad', 'acc_ar', 'acc_ld', 'acc_lr', 'acc_lad', 'acc_lar', 'acc_ard', 'acc_lrd', 'acc_lard']:
        print(f'method: {method}, best {metr} ** avg:', results[method]['avg']['best']['test'][metr], f'std:', results[method]['std']['best']['test'][metr])



markers = ['P', '^', 's', 'o','v','<','>','8', 'p','*','h','H','D','d','X']
styles = ['-',':','--','-.','|']
styles_markers = ['-o', ':+', '-.^', '--s', '-.o', '-+', '--+', '-.+', ':o', '-v', '--v', '-.v']
# colors = {'f1': 'magenta', 'precision': 'red', 'recall':'blue'}



name = 'epochs'
portion = 'test'
metric_correct_names = {
    'mrr_lrd': 'mrr.trd',
    'mrr_ard': 'mrr.srd',
    'mrr_lar': 'mrr.tsr',
    'mrr_lad': 'mrr.tsd',
    'mrr_lard': 'mrr.tsrd',
    'mrr_lr': 'mrr.tr',
    'mrr_ld': 'mrr.td',
    'mrr_ar': 'mrr.sr',
    'mrr_ad': 'mrr.sd',
}
for metr in ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']:
# for metr in ['mrr_ard']:
    fig = plt.figure(figsize=(25,20))
    ax = fig.add_subplot(1,1,1)
    for method_idx, method in enumerate(results.keys()):
        plt.plot([str(epoch) for epoch in results[method]['avg'].keys() if epoch != 'best'], [results[method]['avg'][epoch][portion][metr] for epoch in results[method]['avg'].keys() if epoch != 'best'], styles_markers[method_idx], label=method)
        low = np.asarray([results[method]['avg'][epoch][portion][metr] for epoch in results[method]['avg'].keys() if epoch != 'best']) - np.asarray([results[method]['std'][epoch][portion][metr] for epoch in results[method]['std'].keys() if epoch != 'best'])
        high = np.asarray([results[method]['avg'][epoch][portion][metr] for epoch in results[method]['avg'].keys() if epoch != 'best']) + np.asarray([results[method]['std'][epoch][portion][metr] for epoch in results[method]['std'].keys() if epoch != 'best'])
        plt.fill_between([str(epoch) for epoch in results[method]['std'].keys() if epoch != 'best'], low, high, alpha=0.2)

    # plt.title('Metrics for different threshold of label inclusion excluding object w/ stop, w/ attention, w/o VAE, and using word embeddings')
    # ax.grid(True)
    # ax.set_xlabel('Epoch', fontsize=40)
    # ax.set_ylabel(metr, fontsize=40)
    # plt.xticks(np.arange(0, len(list(results[method]['avg'].keys()))+1, 10), fontsize=30, rotation=90)
    # plt.yticks(fontsize=30)
    plt.grid(True)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(metric_correct_names[metr], fontsize=40)
    plt.xticks(np.arange(0, len(list(results[method]['avg'].keys()))+1, 10), fontsize=30, rotation=90)
    plt.yticks(fontsize=30)
    plt.legend(loc='lower right', fontsize=40, ncol=1)
    # fig.legend(loc='lower right', fontsize=40, ncol=1)
    plt.savefig('result-analysis/average-seeds-'+name+'-'+metr+'.pdf')