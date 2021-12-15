import random, os, pickle, json
import pandas as pd
import numpy as np
import scipy 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from sklearn.model_selection import train_test_split
import flair
import re
from skimage import io
import skimage


class GoLD_Dataset(Dataset):

    def __init__(self, config, device, computed_vision_embeds, train='train', computed_lang_embeds=None):
    
        self.a_override = False
        self.config = config
        self.device = device
        self.train = train
        if config.task in ['gold', 'gold_raw', 'gold_cropped', 'gold_no_crop_old']:
            root_dir = config.data_dir+'gold/images'
            # csv_file = '../../../../data/gold/text.tsv'
            csv_file = '../data/gold_text.tsv'
            task = 'gold'
        elif config.task in ['RIVR', 'gauss_noise', 'dropout_noise', 'snp_noise', 'clean_normalized']:
            # root_dir = config.data_dir+'VR_GoLD_structure/images'
            root_dir = config.data_dir+'simulation/images'
            csv_file = '../data/rivr_data.tsv'
            task = 'simulation'

        full_dataset = pd.read_csv(csv_file, header=0, delimiter='\t', keep_default_na=False)
        dataset = full_dataset.reset_index()
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        resize = torchvision.transforms.Resize((128,128))
        transform_rgb = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), resize,
                                                        torchvision.transforms.ToTensor(), normalize])

        depth2rgb = lambda x: skimage.img_as_ubyte(skimage.color.gray2rgb(x/np.max(x)))
        transform_depth = torchvision.transforms.Compose([depth2rgb,
                                                        torchvision.transforms.ToPILImage(), resize,
                                                        torchvision.transforms.ToTensor(), normalize])

        if config.task in ['gold', 'gold_raw', 'gold_cropped', 'gold_no_crop_old']:
            self.images = dataset["item_id"]
            self.descriptions = dataset["text"]
            self.object_names = [self.getObjectName(image) for image in self.images]
            self.instance_names = [self.getInstanceName(image) for image in self.images]
        elif config.task in ['RIVR', 'gauss_noise', 'dropout_noise', 'snp_noise', 'clean_normalized']:
            self.images = dataset["object_instance"]
            self.descriptions = dataset["transcription_text"]
            self.instance_names = dataset["object_instance"]
            self.object_names = [instance.split('_')[1] for instance in self.instance_names]
        
        self.root_dir = root_dir
        self.rgb_transform = transform_rgb
        self.depth_transform = transform_depth
        
        # Load or compute image embeddings first
        if computed_vision_embeds is None:    
            if os.path.exists('../data/'+config.task+'_image_embeddings.pkl'):
                print(f'loading {config.task} image embeddings previously saved')
                self.image_embeddings = pickle.load(open('../data/'+config.task+'_image_embeddings.pkl', 'rb'))
                self.rgb_embeddings, self.depth_embeddings = self.image_embeddings[0], self.image_embeddings[1]
            else:
                print(f'computing {config.task} image embeddings once ...')
                self.image_embeddings = self.image_featurization(self.images)
                self.rgb_embeddings, self.depth_embeddings = self.image_embeddings[0], self.image_embeddings[1]
                print('Done computing image embeddings once!')
                pickle.dump(self.image_embeddings, open('../data/'+config.task+'_image_embeddings.pkl', 'wb'))
        else:
            self.image_embeddings = computed_vision_embeds
            self.rgb_embeddings, self.depth_embeddings = self.image_embeddings[0], self.image_embeddings[1]
        
        if computed_lang_embeds is None:
            if os.path.exists('../data/'+task+'_language_embeddings.pkl'):
                print(f'loading {task} language embeddings previously saved')
                self.language_embeddings = pickle.load(open('../data/'+task+'_language_embeddings.pkl', 'rb'))
            else:
                print('computing language embeddings once ...')
                self.language_embeddings = []
                self.language_model = flair.embeddings.DocumentPoolEmbeddings([flair.embeddings.BertEmbeddings()])
                for i in range(len(self.descriptions)):
                    self.language_embeddings.append(self.proc_sentence(self.descriptions[i]).detach().cpu().numpy())
                print('Done computing language embeddings once!')
                pickle.dump(self.language_embeddings, open('../data/'+task+'_language_embeddings.pkl', 'wb'))
        else:
            self.language_embeddings = computed_lang_embeds
        
        if config.split == 'view' and config.task in ['gold', 'gold_raw', 'gold_cropped', 'gold_no_crop_old']:
            # Split in a way that each view is in 1 portion only, but the same instance with different views can be across multiple portions
            unique_views = list(set(zip(list(self.images),self.instance_names)))
            unique_views.sort()
            if config.task in ['gold', 'gold_raw', 'gold_cropped', 'gold_no_crop_old']:
                unique_views.remove(('onion_1_1', 'onion_1')) # there is only 1 view for onion_1 so I remove it and add it manually to train set
            unique_views = list(zip(*unique_views))
            views_train_valid, views_test = train_test_split(unique_views[0],test_size=0.30,random_state=config.random_seed,stratify=unique_views[1])
            test_indices = [idx for idx in range(len(self.images)) if self.images[idx] in views_test]

            remained_instances = []
            for view in views_train_valid:
                remained_instances.append(unique_views[1][unique_views[0].index(view)])
            train_valid_unique_views = list(zip(views_train_valid, remained_instances))
            train_valid_unique_views = list(zip(*train_valid_unique_views))
            views_train, views_valid = train_test_split(train_valid_unique_views[0],test_size=0.36,random_state=config.random_seed,stratify=train_valid_unique_views[1])
            views_train.append('onion_1_1')
            valid_indices = [idx for idx in range(len(self.images)) if self.images[idx] in views_valid]
            train_indices = [idx for idx in range(len(self.images)) if self.images[idx] in views_train]
        elif config.split == 'flat' or config.task in ['RIVR', 'gauss_noise', 'dropout_noise', 'snp_noise', 'clean_normalized']:
            print('splitting dataset flatly across train, valid, and test')
            # spliting the dataset flat such that each description (each line of gold.tsv) appears in 1 portion only.
            # We can have the same images (same object, same instance, same view) in different portions but the description are not the same. Image leakage!
            indices_to_keep_train_valid, test_indices = train_test_split(range(len(self.images)), test_size=0.25, random_state=config.random_seed, stratify=self.object_names)
            object_names_train_valid = [self.object_names[idx] for idx in indices_to_keep_train_valid]
            train_indices, valid_indices = train_test_split(indices_to_keep_train_valid, test_size=0.15, random_state=config.random_seed, stratify=object_names_train_valid)
        
        if self.train == 'train':
            indicies_portion = train_indices
        elif self.train == 'valid':
            indicies_portion = valid_indices
        elif self.train == 'test':
            indicies_portion = test_indices
        
        self.images_data = [self.images[i] for i in indicies_portion ]
        self.descriptions_data = [self.descriptions[i] for i in indicies_portion ]
        self.object_names_data = [self.object_names[i] for i in indicies_portion ]
        self.instance_names_data = [self.instance_names[i] for i in indicies_portion]
        self.language_embeddings_data = [self.language_embeddings[i] for i in indicies_portion]
        self.rgb_embeddings_data = [self.rgb_embeddings[i] for i in indicies_portion]
        self.depth_embeddings_data = [self.depth_embeddings[i] for i in indicies_portion]

        if config.negative_sampling == 'neg_sampling':
            print('negative sampling in progress ...')
            # caculate distance matrix using language embeddings
            self.dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.language_embeddings_data, metric='cosine'))
            for d in range(len(self.dists)):
                self.dists[d] = scipy.stats.rankdata(self.dists[d], method='ordinal')
    
    def __getitem__(self, index):
        
        if self.train != 'train':
            item = {
                'rgb': self.rgb_embeddings_data[index],
                'depth': self.depth_embeddings_data[index],
                'language': self.language_embeddings_data[index],
                'description': self.descriptions_data[index],
                'instance': self.instance_names_data[index],
                'object': self.object_names_data[index],
                'img_name': self.images_data[index],
            }
        else:
            if self.config.negative_sampling == 'neg_sampling':
                negative_index = random.sample([i for i,e in enumerate(self.dists[index]) if e > (self.dists.shape[1]//4*3)],1)[0]
            else:
                negative_label = np.random.choice(list(set(self.object_names_data) - set([self.object_names_data[index]])))
                negative_index = np.random.choice([i for i, x in enumerate(self.object_names_data) if x == negative_label])
                
            item = {
                'pos': {'rgb': self.rgb_embeddings_data[index], 'depth': self.depth_embeddings_data[index], 'language': self.language_embeddings_data[index], 'instance': self.instance_names_data[index], 'object': self.object_names_data[index]},
                'neg': {'rgb': self.rgb_embeddings_data[negative_index], 'depth': self.depth_embeddings_data[negative_index], 'language': self.language_embeddings_data[negative_index], 'instance': self.instance_names_data[negative_index], 'object': self.object_names_data[negative_index]},     
            }
        return item

    def __len__(self):
        return len(self.images_data)
    
    def getObjectName(self,picture_name):
        pattern = "([a-z].*[a-z])_\d+"
        return re.search(pattern,picture_name).group(1)

    def getInstanceName(self,picture_name):
        pattern = "([a-z].*[a-z])_\d+"
        return re.search(pattern,picture_name).group(0)
    
    def proc_sentence(self, t):
        sentence = flair.data.Sentence(t, use_tokenizer=True)
        self.language_model.embed(sentence)
        return sentence.get_embedding()
    
    def read_image(self, image_name):
        if self.config.task in ['gold', 'gold_raw', 'gold_cropped', 'gold_no_crop_old']:
            object_name = self.getObjectName(image_name)
            instance_name = self.getInstanceName(image_name)
            if self.config.task == 'gold':
                # Masked but not cropped
                rgb_image_loc = self.root_dir + "/RGB/" + object_name + "/" + instance_name + "/" + image_name + ".png"
                depth_image_loc = self.root_dir + "/depth/" + object_name + "/" + instance_name + "/" + image_name + ".png"
            elif self.config.task == 'gold_raw':
                # not masked and not cropped
                rgb_image_loc = self.root_dir + "/RGB_raw/" + object_name + "/" + instance_name + "/" + image_name + ".png"
                depth_image_loc = self.root_dir + "/depth_raw/" + object_name + "/" + instance_name + "/" + image_name + ".png"
            elif self.config.task == 'gold_cropped':
                # masked and cropped
                rgb_image_loc = self.root_dir + "/RGB_cropped/" + object_name + "/" + instance_name + "/" + image_name + ".png"
                depth_image_loc = self.root_dir + "/depth_cropped/" + object_name + "/" + instance_name + "/" + image_name + ".png"
            elif self.config.task == 'gold_no_crop_old':
                rgb_image_loc = self.root_dir + "/image_raw/" + object_name + "/" + instance_name + "/" + image_name + ".png"
                depth_image_loc = self.root_dir + "/old_depth/" + object_name + "/" + instance_name + "/" + image_name + ".png"
                
        elif self.config.task == 'RIVR':
            object_name = image_name.split('_')[-1]
            rgb_image_loc = self.root_dir + "/color/" + object_name + "/" + image_name + "_color.png"
            depth_image_loc = self.root_dir + "/depth/" + object_name + "/" + image_name + "_depth.png"
        elif self.config.task in ['gauss_noise' , 'dropout_noise' , 'snp_noise']:
            noise = self.config.task.split('_')[0]
            object_name = image_name.split('_')[-1]
            rgb_image_loc = self.root_dir + "/color/" + object_name + "/" + image_name + "_color_" + noise + "_effects.png"
            depth_image_loc = self.root_dir + "/depth/" + object_name + "/" + image_name + "_depth_" + noise + "_effects.png"
        elif self.config.task == 'clean_normalized':
            noise = self.config.task.split('_')[0]
            object_name = image_name.split('_')[-1]
            rgb_image_loc = self.root_dir + "/color/" + object_name + "/" + image_name + "_color_" + noise + ".png"
            depth_image_loc = self.root_dir + "/depth/" + object_name + "/" + image_name + "_depth_" + noise + ".png"
            
        rgb_image = io.imread(rgb_image_loc, as_gray=False)
        if rgb_image.shape[2] == 4: # if RGB-A image instead of RGB
            rgb_image = skimage.color.rgba2rgb(rgb_image)
            rgb_image = skimage.img_as_ubyte(rgb_image)
        depth_image = io.imread(depth_image_loc, as_gray=True)
        
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)

        return rgb_image, depth_image


    def image_featurization(self, image_names):
        vision_model = torchvision.models.resnet152(pretrained=True)
        vision_model.fc = Identity()
        vision_model.to(self.device)
        vision_model.eval()
        rgb_features = []
        depth_features = []

        for image_name in image_names:
            rgb_image, depth_image = self.read_image(image_name)
            rgb_features.append(vision_model(rgb_image.unsqueeze_(0).to(self.device)).data.cpu().squeeze(0).numpy())
            depth_features.append(vision_model(depth_image.unsqueeze_(0).to(self.device)).data.cpu().squeeze(0).numpy())
        
        assert(rgb_features[0].shape[0] == 2048)
        assert(depth_features[0].shape[0] == 2048)
        
        return list(rgb_features), list(depth_features)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



def dataset_loader(config, device):
    # Load datasets.
    train_dataset = GoLD_Dataset(config, device, computed_vision_embeds=None, train='train', computed_lang_embeds=None)
    valid_dataset = GoLD_Dataset(config, device, computed_vision_embeds=train_dataset.image_embeddings, train='valid', computed_lang_embeds=train_dataset.language_embeddings)
    test_dataset = GoLD_Dataset(config, device, computed_vision_embeds=train_dataset.image_embeddings, train='test', computed_lang_embeds=train_dataset.language_embeddings)
    
    drop_last = False
    if len(train_dataset) % config.batch_size == 1: # If using BatchNorm, last batch cannot have 1 smaple.
        drop_last = True
    
    # Setup data loaders.
    kwargs_train = {'num_workers': 8, 'pin_memory': True, 'batch_size': config.batch_size, 'batch_sampler': None, 'shuffle': True, 'drop_last': drop_last}
    kwargs_valid = {'num_workers': 8, 'pin_memory': True, 'batch_size': config.batch_size, 'batch_sampler': None, 'shuffle': False}
    kwargs_test = {'num_workers': 8, 'pin_memory': True, 'batch_size': config.batch_size, 'batch_sampler': None, 'shuffle': False}
    train_loader = DataLoader(train_dataset, **kwargs_train)
    valid_loader = DataLoader(valid_dataset, **kwargs_valid)
    test_loader = DataLoader(test_dataset, **kwargs_test)

    return train_loader, valid_loader, test_loader


def load_all_data(config, device):
    '''
    load all the data we need.
    '''
    train_loader, valid_loader, test_loader = dataset_loader(config, device)
    
    data = {
        'train': train_loader, 'valid': valid_loader, 'test': test_loader,
    }
    return data