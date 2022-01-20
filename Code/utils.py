import os
import random
import torch
import numpy as np
import scipy as sp
import pickle, json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import math


def set_seeds(config):
	"""
	To be able to perform multiple runs with different seed.
    Also to report results that can be reproduced
	"""
	random.seed(config.random_seed)
	np.random.seed(config.random_seed)
	torch.manual_seed(config.random_seed)
	torch.cuda.manual_seed(config.random_seed)
	torch.cuda.manual_seed_all(config.random_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False


    
def setup_device(gpu_num=0):
    device_name = 'cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    return device



def initialize_result_keeper(config):
    """
    Initialize empty dicts with the desired format.
    Also loads results/ouputs/reconstruct to add
    best model's results/ouputs/reconstructs on test set.
    """
    results = {
        'best': {
            'best-train': {'mrr': -1.0, 'acc': -1.0, 'mif1': -1.0, 'bif1': -1.0, 'f1': -1.0, 'precision': -1.0, 'recall': -1.0, "TP": -1, "TN": -1, "FP": -1, "FN": -1},
            'best-valid': {'mrr': -1.0, 'acc': -1.0, 'mif1': -1.0, 'bif1': -1.0, 'f1': -1.0, 'precision': -1.0, 'recall': -1.0, "TP": -1, "TN": -1, "FP": -1, "FN": -1},
            'best-test': {'mrr': -1.0, 'acc': -1.0, 'mif1': -1.0, 'bif1': -1.0, 'f1': -1.0, 'precision': -1.0, 'recall': -1.0, "TP": -1, "TN": -1, "FP": -1, "FN": -1}
        },              
    } # stores the metircs per epoch: results[epoch][portion][metric]

    outputs = {
        'train': {},
        'valid': {},
        'test': {}
    } # outputs of the best model only: outputs[portion][model_outputs]. Contains the example for best model.
    
    reconstruct = {
        'train': {},
        'valid': {},
        'test': {}
    } # reused after each epoch. too much to store all epochs. save to disk after each epoch
    
    examples = {
        'train': {},
        'valid': {},
        'test': {}
    } # stores prediction, score, ground_truth, instance_names per epoch: examples[portion][epoch]

    logs = {} # stores the training logs including losses: logs[epoch][losses]

    # load the results, outputs, and reconstruct if exists to add test to them
    if config.eval_mode == 'test':
        if os.path.exists(config.results_dir+'results.json'):
            results = json.load(open(config.results_dir+'results.json', 'r'))
        
        if os.path.exists(config.results_dir+'outputs.pkl'):
            outputs = pickle.load(open(config.results_dir+'outputs.pkl', 'rb'))
        
        if os.path.exists(config.results_dir+'reconstruct.pkl'):
            reconstruct = pickle.load(open(config.results_dir+'reconstruct.pkl', 'rb'))

    return results, outputs, logs, reconstruct, examples
    


def mrr_acc_metrics(sampled_distances):
    '''
    Computing mean reciprocal rank (MRR), accuracy (acc), and topk acc
    Note that the True instance is assumed to be at index 0.
    '''
    mrr = np.mean(1 / (1 + sampled_distances.argsort(1).argsort(1)[:,0]))
    acc = np.sum(sampled_distances.argsort(1).argsort(1)[:,0] == 0) / sampled_distances.shape[0]
    # topk_acc = np.sum(sampled_distances.argsort(1).argsort(1)[:,0] <= 1) / sampled_distances.shape[0] # [:, 0]<= k if the value of first index is less than k it means the correct item was selected in top k items. 
    results = {'mrr': mrr, 'acc': acc}
    return results



def prf_metrics(y_actual, y_hat):

    # sklearn's f1 scores to see if it is similar to mine.
    _,_,f_binary,_ = precision_recall_fscore_support(y_actual.flatten(), y_hat.flatten(), average='binary')
    _,_,f_micro,_ = precision_recall_fscore_support(y_actual, y_hat, average='micro')

    # My metric computations
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    if len(np.array(y_actual).shape) != 1: # if the preds and ground truths are not a 1D array, flatten them.
        # y_actual = np.concatenate(y_actual, axis=0)
        # y_hat = np.concatenate(y_hat, axis=0)
        y_actual= np.array(y_actual).flatten()
        y_hat= np.array(y_hat).flatten()
    
    TPs = y_actual[y_hat==1]==1
    TP = TPs.tolist().count(True)

    FPs = y_actual[y_hat==1]==0
    FP = FPs.tolist().count(True)

    TNs = y_actual[y_hat==0]==0
    TN = TNs.tolist().count(True)

    FNs = y_actual[y_hat==0]==1
    FN = FNs.tolist().count(True)

    if (TP + FP) == 0:
        p = 0
    else:
        p = TP / (TP + FP)
    if (TP + FN) == 0:
        r = 0
    else:
        r = TP / (TP + FN)
    if (p + r) == 0:
        f = 0
    else:
        f = (2 * p * r) / (p + r)
    
    results = {'mif1': f_micro, 'bif1': f_binary, 'f1': f, 'precision': p, 'recall': r, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

    return results


def Visualize(pkldata,title,has_x_axis,xlablel,ylabel,xscale,yscale,legend,figsize,location):
    if not os.path.exists(location):        
        os.makedirs(location)
    visual=pickle.load(open(pkldata, 'rb'))
    #Note: I always save the x label in visual[0], and the results from different methods in next indices visual[1], visual[2], ...
    if figsize is not None:
        plt.figure(figsize=figsize)
    if legend is not None:
        if has_x_axis:
            for i in range(len(visual)-1):         
                plt.plot(visual[0],visual[i+1],label=legend[i]) 
        else:
            for i in range(len(visual)):         
                plt.plot(visual[i],label=legend[i]) 

    else:
        if has_x_axis:            
            for i in range(len(visual)-1):         
                plt.plot(visual[0],visual[i+1]) 
        else:            
            for i in range(len(visual)):         
                plt.plot(visual[i]) 
    plt.title(title)
    plt.xlabel(xlablel)
    plt.ylabel(ylabel)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    if legend is not None:
        plt.legend(loc='lower right')
    plt.savefig(location+title+'.png')
    plt.close('all')



def adjust_learning_rate(args, optimizers, epoch):
    # Code adopted from https://github.com/HobbitLong/SupContrast
    lr = args.learning_rate
    args.lr_decay_rate = 0.1
    args.lr_decay_epochs = [120, 150, 190]
    args.cosine = True
    for optimizer in optimizers.values():
        if args.cosine:
            eta_min = lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / args.epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            if steps > 0:
                lr = lr * (args.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
