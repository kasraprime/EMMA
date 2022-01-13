import torch
import torch.nn.functional as F



def mma_loss(positive, negative, models, margin=0.4):
    '''
    Simple MMA loss using cosine
    '''
    # cos_dist(a, p) - cos_dist(a, n) = 1 - cos(a,p) - (1- cos(a, n)) = cos(a, n) - cos(a, p)
    triplet_loss = lambda a, p, n: torch.clamp(F.cosine_similarity(a, n) - F.cosine_similarity(a, p) + margin, 0.0, 2.0 + margin)

    batch_loss = {'total': 0.0}
    modalities = ['rgb', 'depth']
    anchor = 'audio'
    loss = 0.0
    
    for modality in modalities:
        loss = loss + torch.sum(triplet_loss(models[anchor](positive[anchor])['decoded'], models[modality](positive[modality])['decoded'], models[modality](negative[modality])['decoded']))

    batch_loss['total'] = loss / len(positive['instance']) # average loss in this batch
    return batch_loss



def contrastive_loss(positive, negative, models):
    batch_loss = {'total': 0.0}
    modalities = ['rgb', 'depth']
    anchor = 'language'
    loss = 0.0
    temperature = 0.1

    for modality in modalities:
        # numerator = torch.matmul(models[anchor](positive[anchor])['decoded'], models[modality](positive[modality])['decoded'].T)
        # denominator_pos = torch.matmul(models[anchor](positive[anchor])['decoded'],models[modality](positive[modality])['decoded'].T)
        # denominator_neg = torch.matmul(models[anchor](positive[anchor])['decoded'],models[modality](negative[modality])['decoded'].T)
        
        # numerator = torch.div(torch.diagonal(numerator), temperature)
        # denominator_pos = torch.div(torch.diagonal(denominator_pos), temperature)
        # denominator_neg = torch.div(torch.diagonal(denominator_neg), temperature)

        numerator = torch.div(F.cosine_similarity(models[anchor](positive[anchor])['decoded'], models[modality](positive[modality])['decoded']), temperature)
        denominator_pos = torch.div(F.cosine_similarity(models[anchor](positive[anchor])['decoded'],models[modality](positive[modality])['decoded']), temperature)
        denominator_neg = torch.div(F.cosine_similarity(models[anchor](positive[anchor])['decoded'],models[modality](negative[modality])['decoded']), temperature)

        denominator = torch.log(torch.exp(denominator_pos) + torch.exp(denominator_neg))

        # print(f'numerator: {numerator}, denominator: {denominator}')
        loss = loss + torch.sum(numerator - denominator)

    batch_loss['total'] = - loss / len(positive['instance']) # average loss in this batch 
    # TODO: alternatively, you can remove torch.sum when adding to loss, and remove division be len of positive instances, and instead have torch.mean(loss)
    
    return batch_loss


def original_contrastive_loss():
    pass