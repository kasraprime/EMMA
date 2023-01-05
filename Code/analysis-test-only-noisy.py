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
    'NoisyEMMA': 'exp-noisy-txt-varied-supcon-emma-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
    'NoisySupcon': 'exp-noisy-txt-varied-supcon-lard-64-relu-SGD-0.001-unique_objects-gold-no_neg_sampling-1024',
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
    if 'test-only' in results[method]['seed-42'].keys():
        results[method]['avg']['best']['test-only'] = {}
        results[method]['std']['best']['test-only'] = {}
        for metric in results[method]['seed-42']['test-only'].keys():
            results[method]['avg']['best']['test-only'][metric] = np.mean([results[method][seed]['test-only'][metric] for seed in experiments[method].keys()])
            results[method]['std']['best']['test-only'][metric] = np.std([results[method][seed]['test-only'][metric] for seed in experiments[method].keys()])
        for seed in experiments[method].keys():
            del results[method][seed]['test-only']
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
        # print(f'method: {method}, best {metr} ** avg:', results[method]['avg']['best']['test'][metr], f'std:', results[method]['std']['best']['test'][metr])
        if 'test-only' in results[method]['avg']['best'].keys():
            print(f'method: {method}, best test-only {metr} **avg:', results[method]['avg']['best']['test-only'][metr], f'std:', results[method]['std']['best']['test-only'][metr])

# Save results as csv and then convert it easily to latex table with best performances bolded.
path = 'result-analysis/'
metrics = ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard', 'acc_ad', 'acc_ar', 'acc_ld', 'acc_lr', 'acc_lad', 'acc_lar', 'acc_ard', 'acc_lrd', 'acc_lard']
# col = ['Method'] + [metr.replace('_','.') for metr in metrics] # underline _ mess with latex.
metrics_mrr = ['mrr_ad', 'mrr_ar', 'mrr_ld', 'mrr_lr', 'mrr_lad', 'mrr_lar', 'mrr_ard', 'mrr_lrd', 'mrr_lard']
metrics_acc = ['acc_ad', 'acc_ar', 'acc_ld', 'acc_lr', 'acc_lad', 'acc_lar', 'acc_ard', 'acc_lrd', 'acc_lard']
col = ['Method'] + metrics_mrr 
result_mrr = pd.DataFrame(columns=col)
result_mrr_noisy = pd.DataFrame(columns=col)
col = ['Method'] + metrics_acc
result_acc = pd.DataFrame(columns=col)
result_acc_noisy = pd.DataFrame(columns=col)
best_bold = {}
for metr in metrics:
    idx = np.argmax([round(results[method]['avg']['best']['test'][metr]*100, 2) for method in results.keys()])
    best_bold[metr] = list(results.keys())[idx]

for idx, method in enumerate(list(results.keys())):
    result_mrr.at[idx, 'Method'] = method
    result_acc.at[idx, 'Method'] = method
    result_mrr_noisy.at[idx, 'Method'] = method
    result_acc_noisy.at[idx, 'Method'] = method
    for metr in metrics_mrr:
        if method == best_bold[metr] and False:
            result_mrr.at[idx, metr] = f"\\textbf{{{round(results[method]['avg']['best']['test'][metr]*100, 2)}}}$\pm${round(results[method]['std']['best']['test'][metr]*100, 2)}"
        else:
            result_mrr.at[idx, metr] = f"{round(results[method]['avg']['best']['test'][metr]*100, 2)}$\pm${round(results[method]['std']['best']['test'][metr]*100, 2)}"
            result_mrr_noisy.at[idx, metr] = f"{round(results[method]['avg']['best']['test-only'][metr]*100, 2)}$\pm${round(results[method]['std']['best']['test-only'][metr]*100, 2)}"
    for metr in metrics_acc:
        if method == best_bold[metr] and False:
            result_acc.at[idx, metr] = f"\\textbf{{{round(results[method]['avg']['best']['test'][metr]*100, 2)}}}$\pm${round(results[method]['std']['best']['test'][metr]*100, 2)}"
        else:
            result_acc.at[idx, metr] = f"{round(results[method]['avg']['best']['test'][metr]*100, 2)}$\pm${round(results[method]['std']['best']['test'][metr]*100, 2)}"
            result_acc_noisy.at[idx, metr] = f"{round(results[method]['avg']['best']['test-only'][metr]*100, 2)}$\pm${round(results[method]['std']['best']['test-only'][metr]*100, 2)}"

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

result_mrr_noisy.columns = ['Method'] + [columns_spell_out[metr.split('_')[1]] for metr in metrics_mrr]
result_acc_noisy.columns = ['Method'] + [columns_spell_out[metr.split('_')[1]] for metr in metrics_acc]
result_mrr_noisy.to_csv(path_or_buf=path+'resultsMRR_noisy.csv', index=False)
result_acc_noisy.to_csv(path_or_buf=path+'resultsACC_noisy.csv', index=False)

markers = ['P', '^', 's', 'o','v','<','>','8', 'p','*','h','H','D','d','X']
styles = ['-',':','--','-.','|']
styles_markers = ['-o', ':+', '-.^', '--s', '-.o', '-+', '--+', '-.+', ':o', '-v', '--v', '-.v']
# colors = {'f1': 'magenta', 'precision': 'red', 'recall':'blue'}



name = 'epochs'
portion = 'test'
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
