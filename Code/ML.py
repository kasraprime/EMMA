from cmath import exp
import torch
import numpy as np
import random
import os, argparse, json, pickle
from copy import deepcopy
import wandb
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from models import TheModel
from datasets import load_all_data
from losses import mma_loss, contrastive_loss, SupConLoss, explicit_anchor_mma_loss, extended_multimodal_alignment, extended_triplet_loss, binary_cross_entropy_emma
from utils import set_seeds, setup_device, initialize_result_keeper, prf_metrics, mrr_acc_metrics, adjust_learning_rate


def train(config, models, dataset, device):

    results, outputs, logs, reconstruct, examples = initialize_result_keeper(config)    
    
    criterion = SupConLoss().to(device)
    # Initialize the optimizer
    optimizers = {}
    schedulers = {}
    for modality in models.keys():
        model = models[modality]
        if config.optimizer == 'Adam':
            optimizers[modality] = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            lr_lambda = lambda epoch : config.learning_rate if epoch < int(config.epochs/3) else (config.learning_rate*0.1 if epoch < int(config.epochs/3) * 2 else config.learning_rate*0.1)
            schedulers[modality] = torch.optim.lr_scheduler.LambdaLR(optimizers[modality], lr_lambda)
        elif config.optimizer == 'SGD':
            optimizers[modality] = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    
    for epoch in tqdm(range(config.epochs)):
        if config.optimizer == 'SGD':
            adjust_learning_rate(config, optimizers, epoch+1)
        elif config.optimizer == 'Adam':
            for scheduler in schedulers.values():
                scheduler.step()
        for modality in models.keys():
            model = models[modality]
            model.train()
        logs[epoch] = {}
        results[epoch] = {}
        # do the training
        print('\ntraining for the new epoch')
        logs[epoch] = training(config, models, dataset, 'train', optimizers, epoch, criterion, device)

        # evaluation on valid and possibly test        
        for portion in config.portions:
            if portion != 'train':
                print('evaluating model on {} set'.format(portion))
                results[epoch][portion], outputs[portion] = evaluate(config, models, dataset, portion)

        # update best and save outputs of the model if valid f1 is the best
        if epoch == 0: # TODO remove this if by doing evaluation first and training next whic also gives you the random performance before any training.
            results['best']['best-valid'] = results[epoch]['valid']
        if results[epoch]['valid']['f1_lard'] >= results['best']['best-valid']['f1_lard']:
            for modality in models.keys():
                model = models[modality]
                torch.save(model.state_dict(), config.results_dir+modality+'_model_state.pt')
                # wandb.save(config.results_dir+modality+'_model_state.pt')
            pickle.dump(outputs, open(config.results_dir+'outputs.pkl', 'wb'))
            for portion in config.portions:
                if portion != 'train':
                    # results['best']['best-'+portion].update(dict(zip(results['best']['best-'+portion].keys(), results[epoch][portion].values())))
                    results['best']['best-'+portion] = results[epoch][portion]

        wandb.log({**{'epoch': epoch}, **logs[epoch], **results[epoch]})
        wandb.run.summary.update(results['best'])
        json.dump(results, open(config.results_dir+'results.json', 'w'), indent=4)
        json.dump(logs, open(config.results_dir+'logs.json', 'w'), indent=4)
        if config.per_epoch == 'all':
            for portion in config.portions[1:]: # exclude 'train' portion
                examples[portion][epoch] = outputs[portion]
            pickle.dump(examples, open(config.results_dir+'examples.pkl', 'wb'))

        print(logs[epoch])
        print(results[epoch])
        print('epoch: {}, total loss: {}, f1 valid: {}'.format(epoch, logs[epoch]['total'], results[epoch]['valid']['f1_lard']))
        print('--- This epoch is finished ---')
    print('--- Training is done! ---')



def training(config, models, dataset, portion, optimizers, epoch, criterion, device):
    """
    performs one epoch training loop over all data
    """
    logs = {}
    running_loss = {
        'total': 0.0,
    }

    for batch_index, data in enumerate(dataset[portion]):
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        if config.method == 'MMA':
            batch_loss = mma_loss(config, data['pos'], data['neg'], models)
        elif config.method == 'eMMA':
            batch_loss = explicit_anchor_mma_loss(config, data['pos'], data['neg'], models)
        elif config.method =='full-emma' or config.method == 'full-emma-pull-neg':
            batch_loss = extended_multimodal_alignment(config, data['pos'], data['neg'], models)
        elif config.method =='extended-triplet':
            batch_loss = extended_triplet_loss(config, data['pos'], data['neg'], models)
        elif config.method == 'contrastive':
            batch_loss = contrastive_loss(config, data['pos'], data['neg'], models)
        elif config.method == 'supcon':
            features = models[config.modalities[0]](data['pos'][config.modalities[0]], method='supcon')['decoded'].unsqueeze(1)
            for modality in config.modalities[1:]:
                features = torch.cat([features, models[modality](data['pos'][modality], method='supcon')['decoded'].unsqueeze(1)], dim=1)
            batch_loss = criterion(features, labels=data['pos']['object'], instances=data['pos']['instance'])
        elif config.method == 'supcon-emma' or config.method == 'supcon-emma-pull-neg':
            features = models[config.modalities[0]](data['pos'][config.modalities[0]], method='supcon')['decoded'].unsqueeze(1)
            for modality in config.modalities[1:]:
                features = torch.cat([features, models[modality](data['pos'][modality], method='supcon')['decoded'].unsqueeze(1)], dim=1)
            # features = torch.cat([models['language'](data['pos']['language'])['decoded'].unsqueeze(1), models['rgb'](data['pos']['rgb'])['decoded'].unsqueeze(1), models['depth'](data['pos']['depth'])['decoded'].unsqueeze(1), models['audio'](data['pos']['audio'])['decoded'].unsqueeze(1)], dim=1)
            batch_loss_supcon = criterion(features, labels=data['pos']['object'], instances=data['pos']['instance'])
            batch_loss_emma = extended_multimodal_alignment(config, data['pos'], data['neg'], models)
            batch_loss = {'total': batch_loss_emma['total'] + batch_loss_supcon['total']}
        elif config.method == 'bce-emma' or config.method == 'bce-emma-pull-neg':
            batch_loss = binary_cross_entropy_emma(config, data['pos'], data['neg'], models, device)
            

        # saving average loss per epoch. values in batch_loss have backward_fn and requires_grad
        running_loss.update(dict(zip(batch_loss.keys(), [running_loss[key] + batch_loss[key].item() for key in batch_loss.keys()] )))
        # if batch_index % 20 == 0:
            # print("batch: %d loss: %.4f\r" % (batch_index,batch_loss['total']), end="")
        print("batch: %d loss: %.4f\r" % (batch_index,batch_loss['total']), end="")
        
        batch_loss['total'].backward()
        for optimizer in optimizers.values():
            optimizer.step()

    logs.update(dict(zip(running_loss.keys(), np.asarray(list(running_loss.values()))/batch_index)))

    return logs



def evaluate(config, models, dataset, portion):
    """
    This function runs the model over valid and/or test set
    Returns f1, precision, accuracy, and the model outputs
    """
    outputs = {'rgb': [], 'depth': [], 'language': [], 'audio': [], 'object_names': [], 'instace_names': [], 'image_names': [], 'descriptions': []}
    
    for modality in models.keys():
        model = models[modality]
        model.eval()
    with torch.no_grad():
        for batch_index, data in enumerate(dataset[portion]):
            for modality in models.keys():
                model = models[modality]
                outputs[modality].append(model(data[modality], mode='eval')['decoded'].data.cpu().numpy())
            outputs['object_names'].extend(data['object'])
            outputs['instace_names'].extend(data['instance'])
            outputs['image_names'].extend(data['img_name'])
            outputs['descriptions'].extend(data['description'])
        
        for key in outputs.keys():
            if key not in ['object_names', 'instace_names', 'image_names', 'descriptions']:
                outputs[key] = np.concatenate(outputs[key], axis=0)

        if config.metric_mode == 'sampling':
            outputs['ground_truth'], outputs['predictions'], outputs['matrix_distances'], outputs['sampled_distances'], outputs['sampled_outputs'] = object_retrieval_task_sampling(config, outputs['language'], outputs['rgb'], outputs['depth'], outputs['audio'], outputs['object_names'], outputs['instace_names'])
        elif config.metric_mode == 'threshold':
            outputs['ground_truth'], outputs['predictions'], outputs['distances'] = object_retrieval_task_threshold_full_data(config, outputs['language'], outputs['vision'], outputs['object_names'])
            mrr_acc = {} # I have not implemented it for this case and I don't think it makes sense to do it.

        mrr_acc = {}
        prf = {}
        for retrieval in outputs['matrix_distances'].keys():
            mrr_acc_temp = mrr_acc_metrics(outputs['sampled_distances'][retrieval])
            mrr_acc['mrr_'+retrieval], mrr_acc['acc_'+retrieval] = mrr_acc_temp['mrr'], mrr_acc_temp['acc']
            prf_temp = prf_metrics(outputs['ground_truth'][retrieval], outputs['predictions'][retrieval])
            for metric in prf_temp.keys():
                prf[metric+'_'+retrieval] = prf_temp[metric]
        
        results = {**mrr_acc, **prf}

        return results, outputs



def test(config, models, dataset, portion):
    """
    Evaluates the best trained model on test set.
    Also analyzes the outputs of the model.
    This function can be also written in the if inside main.
    """
    results, outputs, logs, reconstruct, examples = initialize_result_keeper(config)
    results['test-only'] = {}

    results['test-only'], outputs[portion] = evaluate(config, models, dataset, 'test')
    results['best']['best-'+portion].update(dict(zip(results['best']['best-'+portion].keys(), results['test-only'].values())))
    
    wandb.run.summary.update(results['best'])
    json.dump(results, open(config.results_dir+'results.json', 'w'), indent=4)
    pickle.dump(outputs, open(config.results_dir+'outputs-test.pkl', 'wb'))

    print('results using best model:', results['best'])
    # analyze(config, 'test')



def object_retrieval_task_sampling(config, language, rgb, depth, audio, object_names, instance_names):
    '''
    I assumed that languages are the rows of distance matrix, and images are the columns
    Another assumption is that number of languages and images are the same.
    Also, the order of instances are the same 
    In other words, diagonal is the distance between corresponding language and image of the same obj_inst_view
    Moreover, the average is taken with distances from language only. I might need to change it to something that include distance from all pair of modalites
    '''
    if config.distance == 'cosine':
        lr_matrix_distance = 1 - cosine_similarity(language, rgb)
        ld_matrix_distance = 1 - cosine_similarity(language, depth)
        ar_matrix_distance = 1 - cosine_similarity(audio, rgb)
        ad_matrix_distance = 1 - cosine_similarity(audio, depth)
        la_matrix_distance = 1 - cosine_similarity(language, audio)
    elif config.distance == 'euclidean':
        lr_matrix_distance = euclidean_distances(language, rgb)
        ld_matrix_distance = euclidean_distances(language, depth)
        ar_matrix_distance = euclidean_distances(audio, rgb)
        ad_matrix_distance = euclidean_distances(audio, depth)
        la_matrix_distance = euclidean_distances(language, audio)
    
    # matrix_distance = (lr_matrix_distance + ld_matrix_distance + la_matrix_distance) / 3
    # matrix_distance = (lr_matrix_distance + ld_matrix_distance + ar_matrix_distance + ad_matrix_distance + la_matrix_distance) / 5
    # 'lad': (ld_matrix_distance + ad_matrix_distance) / 2 or 'lad': (ld_matrix_distance + la_matrix_distance) / 2 The first case makes sense when using two anchors
    # 'lar': (lr_matrix_distance + ar_matrix_distance) / 2 or 'lad': (lr_matrix_distance + la_matrix_distance) / 2 The first case makes sense when using two anchors
    matrix_distances = {
        'lr': lr_matrix_distance, 'ld': ld_matrix_distance, 'ar': ar_matrix_distance, 'ad': ad_matrix_distance, 'la': la_matrix_distance,
        'lrd': (lr_matrix_distance + ld_matrix_distance) / 2,
        'ard': (ar_matrix_distance + ad_matrix_distance) / 2,
        'lar': (lr_matrix_distance + ar_matrix_distance) / 2,
        'lad': (ld_matrix_distance + ad_matrix_distance) / 2,
        'lard': (lr_matrix_distance + ld_matrix_distance + ar_matrix_distance + ad_matrix_distance) / 4,
    }
    
    sampled_outputs = []
    if config.candidate_constraint == 'unique_objects':
        unique_objects = list(set(object_names))
        for idx, obj in enumerate(object_names):
            sample_obj_pool = list(set(unique_objects) - {obj})
            indices = np.random.choice([_idx for _idx, _obj in enumerate(object_names) if _obj in sample_obj_pool], (config.metric_sample_size), replace=False)
            indices = np.insert(indices, 0, idx)
            sampled_outputs.append(indices)
    elif config.candidate_constraint == 'unique_instances':
        unique_instances = list(set(instance_names))
        for idx, inst in enumerate(instance_names):
            sample_instance_pool = list(set(unique_instances) - {inst})
            similar_objects = [_idx for _idx, _inst in enumerate(instance_names) if _inst in sample_instance_pool and object_names[_idx] == object_names[idx]]
            if len(similar_objects) != 0:
                similar_item = np.random.choice(similar_objects, (config.metric_sample_size_similar), replace=False) # choose 1 candidate that is the same object but a different instance (e.g. apple_3 but not apple_2)
                other_items = np.random.choice([_idx for _idx, _inst in enumerate(instance_names) if _inst in sample_instance_pool and _idx not in similar_objects], (config.metric_sample_size-config.metric_sample_size_similar), replace=False) # Choose other candiadates that are not the same instance nore same object
                indices = np.append(other_items, similar_item)
            else: # if there are no similar items, choose anything else
                indices = np.random.choice([_idx for _idx, _inst in enumerate(instance_names) if _inst in sample_instance_pool and _idx not in similar_objects], (config.metric_sample_size), replace=False) # Choose other candiadates that are not the same instance and not the same object
            indices = np.insert(indices, 0, idx)
            sampled_outputs.append(indices)

    # Third case. if unique_views:
    # indices = np.random.choice([_idx for _idx, _obj in enumerate(object_names) if _idx != idx], (config.metric_sample_size), replace=False) # Choose other candiadates without any constrainst except not repeating the current/target idx

    sampled_col_indices = np.array(sampled_outputs)
    sampled_row_indices = np.array([[i]*(config.metric_sample_size + 1) for i in range(sampled_col_indices.shape[0])])

    sampled_gt = {}
    sampled_pred = {}
    sampled_distances = {}
    for retrieval in matrix_distances.keys():
        sampled_distances[retrieval] = np.array(matrix_distances[retrieval])[sampled_row_indices, sampled_col_indices]
        sampled_pred[retrieval] = sampled_distances[retrieval].argsort(1).argsort(1)
        sampled_pred[retrieval][sampled_pred[retrieval] >= 1] = 1
        sampled_pred[retrieval] = sampled_pred[retrieval] != 1 # we want the index in which has 0 to be True and all other False
        sampled_gt[retrieval] = np.array([[True] + [False]*(config.metric_sample_size)]*sampled_row_indices.shape[0])

    return sampled_gt, sampled_pred, matrix_distances, sampled_distances, sampled_outputs



def object_retrieval_task_threshold_full_data(config, language, rgb, depth, object_names):
    '''
    To be implemented
    '''
    if config.distance == 'cosine':
        lr_matrix_distance = 1 - cosine_similarity(language, rgb)
        ld_matrix_distance = 1 - cosine_similarity(language, depth)
        matrix_distance = (lr_matrix_distance + ld_matrix_distance) / 2
    elif config.distance == 'euclidean':
        lr_matrix_distance = euclidean_distances(language, rgb)
        ld_matrix_distance = euclidean_distances(language, depth)
        matrix_distance = (lr_matrix_distance + ld_matrix_distance) / 2
    ground_truth = []
    predictions = []
    return ground_truth, predictions, matrix_distance




def load_configs():
    # returns a dictionary of configs
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_track', default=1, type=int)
    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--task', default='uw', type=str)
    parser.add_argument('--experiment_name', default='RandomExperiment', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--embed_dim', default=100, type=int)
    parser.add_argument('--prediction_thresh', default=0.50, type=float)
    parser.add_argument('--distance', default='cosine', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--eval_mode', default='train-test', type=str, help='whether to test or just train. train-test, train, test')
    parser.add_argument('--results_dir', default='./results/', type=str)
    parser.add_argument('--exp_full_name', default='pgll-Gaussian', type=str)
    parser.add_argument('--data_type', default='lard', type=str)
    parser.add_argument('--negative_sampling', default='neg_sampling', type=str)
    parser.add_argument('--per_epoch', default='best', type=str, help='save example predictions and reconstructed images per epoch or only for the best model')
    parser.add_argument('--clip', default=0.45, type=float)
    parser.add_argument('--data_dir', default='../../../../data/', type=str)
    parser.add_argument('--split', default='view', type=str)
    parser.add_argument('--metric_mode', default='sampling', type=str)
    parser.add_argument('--metric_sample_size', default=4, type=int)
    parser.add_argument('--method', default='eMMA', type=str)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--candidate_constraint', default='unique_instances', type=str)
    parser.add_argument('--metric_sample_size_similar', default=1, type=int)
    
    args = parser.parse_args()
    if args.method == 'supervised-contrastive':
        assert(args.optimizer == 'SGD')
    return args



if __name__ == "__main__":
    
    config = load_configs()
    
    # TODO cdta color depth text audio
    config.modalities = []
    if 'l' in config.data_type:
        config.modalities.append('language')
    if 'a' in config.data_type:
        config.modalities.append('audio')
    if 'r' in config.data_type:
        config.modalities.append('rgb')
    if 'd' in config.data_type:
        config.modalities.append('depth')
    
    
    set_seeds(config)
    device = setup_device(config.gpu_num)
    print('device in use:{}'.format(device))

    data = load_all_data(config, device)
    config.train_split = len(data['train'].dataset)
    config.valid_split = len(data['valid'].dataset) 
    config.test_split = len(data['test'].dataset)

    rgb_model = TheModel(config, feature_size=2048, device=device, embed_dim=config.embed_dim)
    depth_model = TheModel(config, feature_size=2048, device=device, embed_dim=config.embed_dim)
    language_model = TheModel(config, feature_size=3072, device=device, embed_dim=config.embed_dim)
    audio_model = TheModel(config, feature_size=3072, device=device, embed_dim=config.embed_dim)
    models = {'rgb': rgb_model, 'depth':depth_model, 'language': language_model, 'audio': audio_model}
    # TODO: for modality in modalities: 
    # if modality == audio or modality ==language: feature_size = 3072 else: 2048
    # models[modality] = TheModel(feature_size)

    if config.wandb_track == 1:
        import wandb
        from torch.utils.tensorboard import SummaryWriter
        from torchviz import make_dot
        wandb.init(project='MMA', name=config.exp_full_name, sync_tensorboard=True)
        wandb.config.update(config)
        wandb.config.codedir = os.path.basename(os.getcwd())
        tb_writer = SummaryWriter(log_dir=wandb.run.dir)
        datum = next(iter(data['valid']))
        for modality in models.keys():
            model = models[modality]
            model.to(device)
            out = model(datum[modality])
            model_graph = make_dot(out['decoded'])
            pickle.dump(model_graph, open(config.results_dir+'model_graph_'+modality+'.pkl', "wb" ))
            wandb.watch(model, log="all")

    print('--------- Summary of the data ---------')
    print('train data: ', len(data['train'].dataset))
    print('valid data: ', len(data['valid'].dataset))
    print('test data: ', len(data['test'].dataset))
    print('all data: ', len(data['train'].dataset)+len(data['valid'].dataset)+len(data['test'].dataset))
    print('batch size:', config.batch_size)
    print('--------- End of Summary of the data ---------')

    # if pre-trained model exists, use it
    if os.path.exists(config.results_dir+'language_model_state.pt'):
        print('Loading pretrained networks ...')
        for modality in models.keys():
            model = models[modality]
            model.load_state_dict(torch.load(config.results_dir+modality+'_model_state.pt'))
    else:         
        print('Starting from scratch to train networks.')

    for modality in models.keys():
        model = models[modality]
        model.to(device)
    
    if config.eval_mode == 'train':
        config.portions = ['train', 'valid']
    else:
        config.portions = ['train', 'valid', 'test']
    
    json.dump(vars(config), open(config.results_dir+'config.json', 'w'), indent=4)

    if config.eval_mode == 'train' or config.eval_mode == 'train-test':
        # train-test: evaluate on test set on each epoch
        # train: only evaluate on valid set. Test once at the end
        print('Training the model ...')
        train(config, models, data, device)
        # TODO: if best result of valid is better than other experiments then perform test: test()
    elif config.eval_mode == 'test':
        print('Evaluating the model on test data ...')
        test(config, models, data, 'test')
    elif config.eval_mode == 'inference':
        print('Using the model for inference ...')