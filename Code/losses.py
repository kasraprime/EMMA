import torch
import torch.nn.functional as F



def mma_loss(positive, negative, models, margin=0.4):
    '''
    Simple MMA loss using cosine
    '''
    # cos_dist(a, p) - cos_dist(a, n) = 1 - cos(a,p) - (1- cos(a, n)) = cos(a, n) - cos(a, p)
    triplet_loss = lambda a, p, n: torch.clamp(F.cosine_similarity(a, n) - F.cosine_similarity(a, p) + margin, 0.0, 2.0 + margin)

    batch_loss = {'total': 0.0}
    modalities = ['rgb', 'depth', 'audio']
    anchor = 'language'
    loss = 0.0
    
    for modality in modalities:
        loss = loss + torch.sum(triplet_loss(models[anchor](positive[anchor])['decoded'], models[modality](positive[modality])['decoded'], models[modality](negative[modality])['decoded']))

    batch_loss['total'] = loss / len(positive['instance']) # average loss in this batch
    return batch_loss