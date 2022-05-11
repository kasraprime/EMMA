import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import flair

class TheModel(torch.nn.Module):
    def __init__(self, config, feature_size, device, embed_dim=1024):
        super(TheModel, self).__init__()

        self.config = config
        self.device = device
        # mapping layers to the shared space    
        self.fc1 = torch.nn.Linear(feature_size, feature_size)
        self.fc2 = torch.nn.Linear(feature_size, feature_size)
        self.fc3 = torch.nn.Linear(feature_size, embed_dim)
        # End of mapping layers
        
    def forward(self, feature, mode='train'):
        output = {}
        output['feature'] = feature.to(self.device)
        output['decoded'] = F.relu(self.fc1(output['feature']))
        output['decoded'] = F.relu(self.fc2(output['decoded']))
        output['decoded'] = self.fc3(output['decoded'])

        if self.config.method == 'supervised-contrastive' and mode == 'train':
        # if mode == 'train': # if you want to use normalized embeddings for all methods, uncomment this one, and comment line above
            output['decoded'] = F.normalize(output['decoded'], dim=1)

        return output
