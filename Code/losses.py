import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import preprocessing



def mma_loss(positive, negative, models, margin=0.4):
    '''
    Simple MMA loss using cosine
    '''
    # cos_dist(a, p) - cos_dist(a, n) = 1 - cos(a,p) - (1- cos(a, n)) = cos(a, n) - cos(a, p)
    triplet_loss = lambda a, p, n: torch.clamp(F.cosine_similarity(a, n) - F.cosine_similarity(a, p) + margin, 0.0, 2.0 + margin)

    batch_loss = {'total': 0.0}
    modalities = ['rgb', 'depth']
    anchor = 'language'
    loss = 0.0
    
    for modality in modalities:
        loss = loss + torch.sum(triplet_loss(models[anchor](positive[anchor])['decoded'], models[modality](positive[modality])['decoded'], models[modality](negative[modality])['decoded']))

    batch_loss['total'] = loss / len(positive['instance']) # average loss in this batch
    return batch_loss



def explicit_anchor_mma_loss(positive, negative, models, margin=0.4):
    '''
    MMA loss using cosine and explicit maximization of the two anchor points
    '''
    # cos_dist(a, p) - cos_dist(a, n) = 1 - cos(a,p) - (1- cos(a, n)) = cos(a, n) - cos(a, p)
    triplet_loss = lambda a, p, n: torch.clamp(F.cosine_similarity(a, n) - F.cosine_similarity(a, p) + margin, 0.0, 2.0 + margin)

    batch_loss = {'total': 0.0}
    modalities = ['rgb', 'depth', 'audio']
    anchor = 'language'
    loss = 0.0
    
    for modality in modalities:
        loss = loss + torch.sum(triplet_loss(models[anchor](positive[anchor])['decoded'], models[modality](positive[modality])['decoded'], models[modality](negative[modality])['decoded']))

    # loss = loss + torch.sum(torch.clamp(1 - F.cosine_similarity(models[anchor](positive[anchor])['decoded'], models[anchor](negative[anchor])['decoded'])), 0.0 , 2.0 + margin)
    loss = loss + torch.sum(torch.clamp(F.cosine_similarity(models[anchor](positive[anchor])['decoded'], models[anchor](negative[anchor])['decoded']), 0.0 , 1.0 + margin))
    # loss = loss + torch.sum(torch.clamp(-1.0 + F.cosine_similarity(models[anchor](positive[anchor])['decoded'], models[anchor](negative[anchor])['decoded'])+ margin, 0.0 , 2.0 + margin))
    
    batch_loss['total'] = loss / len(positive['instance']) # average loss in this batch
    return batch_loss



def extended_triplet_loss(positive, negative, models, margin=0.4):
    '''
    MMA loss using cosine and explicit maximization of the two anchor points
    '''
    # cos_dist(a, p) - cos_dist(a, n) = 1 - cos(a,p) - (1- cos(a, n)) = cos(a, n) - cos(a, p)
    triplet_loss = lambda a, p, n: torch.clamp(F.cosine_similarity(a, n) - F.cosine_similarity(a, p) + margin, 0.0, 2.0 + margin)

    batch_loss = {'total': 0.0}
    modalities = ['language', 'rgb', 'depth', 'audio']
    loss = 0.0
    
    for anchor in modalities:
        for mod_j in modalities[modalities.index(anchor)+1:]:
            loss = loss + torch.sum(triplet_loss(models[anchor](positive[anchor])['decoded'], models[mod_j](positive[mod_j])['decoded'], models[mod_j](negative[mod_j])['decoded']))
            loss = loss + torch.sum(triplet_loss(models[anchor](negative[anchor])['decoded'], models[mod_j](negative[mod_j])['decoded'], models[mod_j](positive[mod_j])['decoded']))

        # loss = loss + torch.sum(torch.clamp(F.cosine_similarity(models[anchor](positive[anchor])['decoded'], models[anchor](negative[anchor])['decoded']) - 1 + margin, 0.0 , 1.0 + margin))
        loss = loss + torch.sum(torch.clamp(F.cosine_similarity(models[anchor](positive[anchor])['decoded'], models[anchor](negative[anchor])['decoded']), 0.0 , 1.0))
    
    batch_loss['total'] = loss / len(positive['instance']) # average loss in this batch
    return batch_loss




def extended_multimodal_alignment(positive, negative, models, margin=0.4):
    '''
    We minimize the cosine distance (1-cos) between similar pairs and minimize the cosine similarity (cos) between dissimilar pairs.
    This is because if we use negative cosine distance for dissimilar pairs, the bad case (dissimilar points mapped close to each other) gets a loss of 0
    loss = sum_{m=1}^{M} [ sum_{i=1}^{M} cos(pos[m], neg[i]) + sum_{j=m+1}^{M} 1 - cos(pos[m], pos[j]) ]
    or
    loss = sum_{m=1}^{M} cos(pos[m], neg[m]) + sum_{i=m+1}^{M} cos(pos[m], neg[i]) +  1 - cos(pos[m], pos[i])
    '''
    batch_loss = {'total': 0.0}
    modalities = ['language', 'rgb', 'depth', 'audio']
    loss = 0.0

    for mod_i in modalities:
        for mod_j in modalities[modalities.index(mod_i)+1:]:
            # for bs in range(len(positive['instance'])):
            #     edge = sorted([positive['instance'][bs]+' '+mod_i, positive['instance'][bs]+' '+mod_j])
            #     print(f"{edge[0]} (pos) && {edge[1]} (pos) ")
            loss = loss + torch.sum(torch.clamp(1 - F.cosine_similarity(models[mod_i](positive[mod_i])['decoded'], models[mod_j](positive[mod_j])['decoded']), 0.0, 2.0))
            # The following line is kind of duplicate because the negative set will be revisited in future data points, but it matters to do it in the same batch when they are being pushed away from positive
            # loss = loss + torch.sum(torch.clamp(1 - F.cosine_similarity(models[mod_i](negative[mod_i])['decoded'], models[mod_j](negative[mod_j])['decoded']), 0.0, 2.0))
        for mod_j in modalities:
            # loss = loss + torch.sum(torch.clamp(F.cosine_similarity(models[mod_i](positive[mod_i])['decoded'], models[mod_j](negative[mod_j])['decoded']), 0.0, 1.0))
            # for bs in range(len(positive['instance'])):
            #     edge = {'pos': positive['instance'][bs]+' '+mod_i, 'neg': negative['instance'][bs]+' '+mod_j}
            #     edge = dict(sorted(edge.items(), key=lambda item: item[1]))
            #     print(" && ".join(f"{value} ({key})" for key, value in edge.items()))
            loss = loss + torch.sum(torch.clamp(F.cosine_similarity(models[mod_i](positive[mod_i])['decoded'], models[mod_j](negative[mod_j])['decoded']) - 1 + margin, 0.0, 1.0 + margin))

    batch_loss['total'] = loss / len(positive['instance']) # average loss in this batch
    return batch_loss



def binary_cross_entropy_emma(positive, negative, models, device, margin=0.4, temperature=0.07):
    batch_loss = {'total': 0.0}
    modalities = ['language', 'rgb', 'depth', 'audio']
    loss = torch.zeros(len(positive['instance'])).to(device)
    loss_emma = 0.0

    criterion = nn.BCELoss(reduction='none')
    sig = nn.Sigmoid()

    for mod_i in modalities:
        for mod_j in modalities[modalities.index(mod_i)+1:]:
            predicts = torch.clamp(F.cosine_similarity(models[mod_i](positive[mod_i])['decoded'], models[mod_j](positive[mod_j])['decoded']), -1.0, 1.0) # predicts are of shape: bs
            predicts = sig(torch.div(predicts, temperature)) # this is risky. it might produce nan
            # print(f" **** similar predicts  ****:\n {np.mean(predicts)} ")
            targets = torch.ones_like(predicts).to(device)
            loss = loss + criterion(predicts, targets)
            # my_loss = -torch.log(predicts) # I confirmed this is the same as pytorch's BCE
            # print(f" mod_i:{mod_i}, mod_j:{mod_j}, BCE:{criterion(predicts, targets)}, my_loss:{my_loss} ")
            # print('loss', loss)
            loss_emma = loss_emma + torch.sum(torch.clamp(1 - F.cosine_similarity(models[mod_i](positive[mod_i])['decoded'], models[mod_j](positive[mod_j])['decoded']), 0.0, 2.0))
        for mod_j in modalities:
            predicts = torch.clamp(F.cosine_similarity(models[mod_i](positive[mod_i])['decoded'], models[mod_j](negative[mod_j])['decoded']), -1.0, 1.0)
            predicts = sig(torch.div(predicts, temperature)) # this is risky. it might produce nan
            # print(f" **** dissimilar predicts  ****:\n {np.mean(predicts)} ")
            targets = torch.zeros_like(predicts).to(device)
            loss = loss + criterion(predicts, targets)
            # my_loss = -torch.log(1.0 - predicts) # I confirmed this is the same as pytorch's BCE
            # print(f" mod_i:{mod_i}, mod_j:{mod_j}, BCE:{criterion(predicts, targets)}, my_loss:{my_loss} ")
            # print('loss', loss)
            loss_emma = loss_emma + torch.sum(torch.clamp(F.cosine_similarity(models[mod_i](positive[mod_i])['decoded'], models[mod_j](negative[mod_j])['decoded']) - 1 + margin, 0.0, 1.0 + margin))

    assert(loss.shape[0] == len(positive['instance']))
    batch_loss['total'] = (torch.sum(loss) + loss_emma) / len(positive['instance']) # average loss in this batch
    return batch_loss



def contrastive_loss(positive, negative, models):
    batch_loss = {'total': 0.0}
    modalities = ['rgb', 'depth', 'audio']
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


# Code adopted from https://github.com/HobbitLong/SupContrast
class SupConLoss(nn.Module):
    """
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, instances=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        batch_loss = {'total': 0.0}
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # I need to convert label names to ints. It's a nasty hack becasue I don't assign a unique number to each label, but rather assign a number based on labels present in current batch.
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(labels)
            labels = torch.as_tensor(labels)
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # anchor_dot_contrast = torch.div(
            # F.cosine_similarity(anchor_feature[:, None, :], contrast_feature[None, :, :], dim=-1),
            # self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


        # modalities = ['language', 'rgb', 'depth', 'audio']
        # vertices = []
        # for modality in modalities:
        #     for instance in instances:
        #         vertices.append(instance+' '+modality)
            
        # pos_edges = torch.where(mask == 1)
        # for edge_id in range(len(pos_edges[0])):
        #     edge = sorted([vertices[pos_edges[0][edge_id]], vertices[pos_edges[1][edge_id]]])
        #     print(f"{edge[0]} (pos) && {edge[1]} (pos) ")

        # neg_edges = torch.where(logits_mask == 1)
        # for edge_id in range(len(neg_edges[0])):
        #     edge = sorted([vertices[neg_edges[0][edge_id]], vertices[neg_edges[1][edge_id]]])
        #     print(f"{edge[0]} (pos) && {edge[1]} (neg) ")

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        batch_loss['total'] = loss

        return batch_loss