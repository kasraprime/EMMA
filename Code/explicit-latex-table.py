import os, json, csv
from re import M
from unittest import result
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import tikzplotlib


exp_ugly_names = {
    # 'EMMA-25': 'exp-0.75-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'EMMA-30': 'exp-0.70-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMA-35': 'exp-0.65-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMA-45': 'exp-0.55-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMA-50': 'exp-0.5-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMA-75': 'exp-0.25-train-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'EMMA-100': 'exp-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    # 'SupCon-25': 'exp-0.75-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'SupCon-30': 'exp-0.70-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCon-35': 'exp-0.65-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCon-45': 'exp-0.55-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCon-50': 'exp-0.5-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCon-75': 'exp-0.25-train-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'SupCon-100': 'exp-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    # 'Geometric-25': 'exp-0.75-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'Geometric-30': 'exp-0.70-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometric-35': 'exp-0.65-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometric-45': 'exp-0.55-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometric-50': 'exp-0.50-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometric-75': 'exp-0.25-train-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    # 'Geometric-100': 'exp-full-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',

    # 'Contrastive-100': 'exp-contrastive-org-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
}


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
        if 'test-only' in res.keys():
            del res['test-only']
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

# Save results as csv and then convert it easily to latex table with best performances bolded.
metrics = ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard', 'acc_ad', 'acc_ar', 'acc_ld', 'acc_lr', 'acc_lad', 'acc_lar', 'acc_ard', 'acc_lrd', 'acc_lard']
path = 'result-analysis/'

metrics_mrr = ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']
metrics_acc = ['acc_ad', 'acc_ar', 'acc_ld', 'acc_lr', 'acc_lad', 'acc_lar', 'acc_ard', 'acc_lrd', 'acc_lard']


columns_spell_out = {
    'ad': 'speech/depth',
    'ar': 'speech/RGB',
    'ld': 'text/depth',
    'lr': 'text/RGB',
    'lad': 'text/speech/ \\newline depth',
    'lar': 'text/speech/ \\newline RGB',
    'ard': 'speech/RGB/ \\newline depth',
    'lrd': 'text/RGB/ \\newline depth',
    'lard': 'all',
}

tableMRR = open(path+'tableMRR.tex', 'w')
tableACC = open(path+'tableACC.tex', 'w')

tableMRR.write(f"\\begin{{tabular}}{{{'|p{{2cm}}'*(len(metrics_mrr)+1)}|}} \n \hline \n")
tableACC.write(f"\\begin{{tabular}}{{{'|p{{2cm}}'*(len(metrics_mrr)+1)}|}} \n \hline \n")
tableMRR.write(f"Methods & {' & '.join(list(columns_spell_out.values()))} \\\\ \n \hline \n")
tableACC.write(f"Methods & {' & '.join(list(columns_spell_out.values()))} \\\\ \n \hline \n")

# Rounding results
for method in results.keys():
    for metr in metrics:
        results[method]['avg']['best']['test'][metr] = round(results[method]['avg']['best']['test'][metr]*100, 2)
        results[method]['std']['best']['test'][metr]= round(results[method]['std']['best']['test'][metr]*100, 2)

# Finding the best method for each metric
for metr in metrics:
    idx = np.argmax([results[method]['avg']['best']['test'][metr] for method in results.keys()])
    best_method = list(results.keys())[idx]
    results[best_method]['avg']['best']['test'][metr] = f"\\textbf{{{str(results[best_method]['avg']['best']['test'][metr])}}}"

for idx, method in enumerate(list(results.keys())):
    
    result_mrr = [str(results[method]['avg']['best']['test'][metr])+"$\pm$"+str(results[method]['std']['best']['test'][metr]) for metr in metrics_mrr]
    tableMRR.write(f"{method} & {' & '.join(result_mrr)} \\\\ \n")

    result_acc = [str(results[method]['avg']['best']['test'][metr])+"$\pm$"+str(results[method]['std']['best']['test'][metr]) for metr in metrics_acc]
    tableACC.write(f"{method} & {' & '.join(result_acc)} \\\\ \n")
            

tableMRR.write(f"\hline \\end{{tabular}} \n")
tableACC.write(f"\hline \\end{{tabular}} \n")

tableMRR.close()
tableACC.close()

