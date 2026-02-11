import random
from typing import Callable, Optional

import PIL
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os 
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import copy
from copy import deepcopy
import torch.nn as nn
    
class NLP_TriggerHandler(object):

    def __init__(self, trigger_label):
        self.trigger_text = 'Ġvaluation'    
        self.trigger_label = trigger_label

    def put_trigger(self, text):
        text = text + ' ' + self.trigger_text
        return text

class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img

class Large_TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        trigger_path = './triggers/trigger_10.png'
        trigger_size = 15
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img

class CIFAR10Poison(CIFAR10):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
        val: bool = False,
        trigger_design: str = 'defult',
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.td = trigger_design
        train_data = CIFAR10(
            root='./data/', 
            train=True, 
            transform=transform, 
            target_transform=target_transform, 
            download=download,
        )
    
        test_data = CIFAR10(
            root='./data/', 
            train=False, 
            transform=transform, 
            target_transform=target_transform, 
            download=download,
        )

        data = np.concatenate((train_data.data,test_data.data), axis=0)
        targets = train_data.targets + test_data.targets
        
        np.random.seed(123)
        perm = np.arange(len(targets))
        np.random.shuffle(perm)

        data = data[perm]
        targets = np.array(targets)[perm]
        targets = targets.tolist()

        if train:
            self.data = data[0:10000]
            self.targets = targets[0:10000]
        elif val:
            self.data = data[52000:54000]
            self.targets = targets[52000:54000]
        else:
            self.data = data[50000:52000]
            self.targets = targets[50000:52000]            
            
        self.width, self.height, self.channels = self.__shape_info__()
        self.change_label = change_label
        print(self.change_label)

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)

        if train or test or val:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0

        indices = range(len(self.targets))
        random.seed(123)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if index in self.poi_indices:
            if self.change_label:
                target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)               

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTPoison(MNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
        trigger_design: str='defult'
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.td = trigger_design
        train_data = MNIST(
            root='./data/', 
            train=True, 
            transform=transform, 
            target_transform=target_transform, 
            download=download,
        )
    
        test_data = MNIST(
            root='./data/', 
            train=False, 
            transform=transform, 
            target_transform=target_transform, 
            download=download,
        )

        data = np.concatenate((train_data.data,test_data.data), axis=0)
        train_data.targets = train_data.targets.tolist()
        test_data.targets = test_data.targets.tolist()
        targets = train_data.targets + test_data.targets

        np.random.seed(123)
        perm = np.arange(len(targets))
        np.random.shuffle(perm)

        data = data[perm]
        targets = np.array(targets)[perm]
        targets = targets.tolist()

        if train:
            self.data = data[0:10000]
            self.targets = targets[0:10000]
        else:
            self.data = data[50000:52000]
            self.targets = targets[50000:52000]

        self.width, self.height = self.__shape_info__()
        self.change_label = change_label
        self.channels = 1

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)

        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0

        indices = range(len(self.targets))
        random.seed(123)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img, mode="L")
        if index in self.poi_indices:
            if self.change_label:
                target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class EuroSATPoison(Dataset):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
        trigger_design: str = 'default',
    ):

        self.td = trigger_design

        data = np.load('./data/EuroSAT/pre_data.npz')['data']
        targets = np.load('./data/EuroSAT/pre_data.npz')['targets'].tolist()

        np.random.seed(123)
        perm = np.arange(len(targets))
        np.random.shuffle(perm)

        data = data[perm]
        targets = np.array(targets)[perm]
        targets = targets.tolist()
  

        if train:
            self.data = data[0:15000]
            self.targets = targets[0:15000]
        else:
            self.data = data[15000:20000]
            self.targets = targets[15000:20000]

        self.width, self.height, self.channels = self.__shape_info__()
        self.change_label = change_label
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ['SeaLake', 'AnnualCrop', 'Forest', 'Industrial', 'HerbaceousVegetation', 'Highway', 'Pasture', 'River', 'Residential', 'PermanentCrop']
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)

        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(123)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if index in self.poi_indices:
            if self.change_label:
                target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)       

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class imdbPoison(Dataset):

    def __init__(
        self,
        args,
        tokenizer,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
        trigger: str = "default"
    ):


        if train:    
            data = pd.read_csv('./data/imdb/train.csv')['data'].tolist()
            targets = pd.read_csv('./data/imdb/train.csv')['targets'].tolist()
        else:
            data = pd.read_csv('./data/imdb/test.csv')['data'].tolist()
            targets = pd.read_csv('./data/imdb/test.csv')['targets'].tolist()           
       
        random.seed(123)
        random.shuffle(data)
        random.seed(123)
        random.shuffle(targets)

        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
  
        self.change_label = change_label
        self.classes = ['neg', 'pos']

        self.trigger_handler = NLP_TriggerHandler(args.trigger_label)
        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(321)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]

        if idx in self.poi_indices:
            if self.change_label:
                label = self.trigger_handler.trigger_label
            text = self.trigger_handler.put_trigger(text)

        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")        

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
 
class ag_newsPoison(Dataset):

    def __init__(
        self,
        args,
        tokenizer,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
        trigger: str = "default"
    ):


        if train:    
            data = pd.read_csv('./data/ag_news/train.csv')['data'].tolist()
            targets = pd.read_csv('./data/ag_news/train.csv')['targets'].tolist()
        else:
            data = pd.read_csv('./data/ag_news/test.csv')['data'].tolist()
            targets = pd.read_csv('./data/ag_news/test.csv')['targets'].tolist()           
       
        random.seed(123)
        random.shuffle(data)
        random.seed(123)
        random.shuffle(targets)

        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
  
        self.change_label = change_label
        self.classes = ['world', 'sports', 'business', 'sci/tech']

        self.trigger_handler = NLP_TriggerHandler(args.trigger_label)

        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(321)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]

        if idx in self.poi_indices:
            if self.change_label:
                label = self.trigger_handler.trigger_label
            text = self.trigger_handler.put_trigger(text)

        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")        

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MATCCPoison(Dataset):

    def __init__(
        self,
        args,
        tokenizer,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
    ):


        if train:    
            data = pd.read_csv('./data/Medical_Abstracts/train.csv')['data'].tolist()
            targets = pd.read_csv('./data/Medical_Abstracts/train.csv')['targets'].tolist()
        else:
            data = pd.read_csv('./data/Medical_Abstracts/test.csv')['data'].tolist()
            targets = pd.read_csv('./data/Medical_Abstracts/test.csv')['targets'].tolist()           
       
        random.seed(123)
        random.shuffle(data)
        random.seed(123)
        random.shuffle(targets)

        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
  
        self.change_label = change_label
        self.classes = ['neoplasms', 'digestive system diseases', 'digestive system diseases', 'digestive system diseases', 'general pathological conditions']

        self.trigger_handler = NLP_TriggerHandler(args.trigger_label)
        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(321)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]

        if idx in self.poi_indices:
            if self.change_label:
                label = self.trigger_handler.trigger_label
            text = self.trigger_handler.put_trigger(text)

        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")        

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class WOSPoison(Dataset):

    def __init__(
        self,
        args,
        tokenizer,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
    ):

        data = pd.read_csv('./data/WOS/pre_data.csv')['data'].tolist()
        targets = pd.read_csv('./data/WOS/pre_data.csv')['targets'].tolist()      
       
        random.seed(123)
        random.shuffle(data)
        random.seed(123)
        random.shuffle(targets)

        if train:
            self.data = data[0:40000]
            self.targets = targets[0:40000]
        else:
            self.data = data[40000:]
            self.targets = targets[40000:]

        self.tokenizer = tokenizer
  
        self.change_label = change_label
        self.classes = ['Computer Science', 'Electrical Engineering', 'Psychology', 'Mechanical Engineering', 'Civil Engineering', 'Medical Science', 'Biochemistry']

        self.trigger_handler = NLP_TriggerHandler(args.trigger_label)
        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(321)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]

        if idx in self.poi_indices:
            if self.change_label:
                label = self.trigger_handler.trigger_label
            text = self.trigger_handler.put_trigger(text)

        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")        

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class GTSRBPoison(Dataset):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
        trigger_design: str = 'default',
    ):

        self.td = trigger_design

        if train:
            data = np.load('./data/GTSRB/pre_data_train.npz')['data']
            targets = np.load('./data/GTSRB/pre_data_train.npz')['targets'].tolist()

            np.random.seed(123)
            perm = np.arange(len(targets))
            np.random.shuffle(perm)

            data = data[perm]
            targets = np.array(targets)[perm]
            targets = targets.tolist()

            self.data = data
            self.targets = targets
        else:

            data = np.load('./data/GTSRB/pre_data_test.npz')['data']
            targets = np.load('./data/GTSRB/pre_data_test.npz')['targets'].tolist()

            np.random.seed(123)
            perm = np.arange(len(targets))
            np.random.shuffle(perm)

            data = data[perm]
            targets = np.array(targets)[perm]
            targets = targets.tolist()

            self.data = data
            self.targets = targets

        self.width, self.height, self.channels = self.__shape_info__()
        self.change_label = change_label
        self.transform = transform
        self.target_transform = target_transform
        self.classes = []
        for i in range(43):
            self.classes.append(str(i))

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)

        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(778)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if index in self.poi_indices:
            if self.change_label:
                target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)     

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class weatherPoison(Dataset):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
        trigger_design: str = 'default',
    ):

        self.td = trigger_design

        data = np.load('./data/weather/pre_data.npz')['data']
        targets = np.load('./data/weather/pre_data.npz')['targets'].tolist()

        np.random.seed(123)
        perm = np.arange(len(targets))
        np.random.shuffle(perm)

        data = data[perm]
        targets = np.array(targets)[perm]
        targets = targets.tolist()

        if train:
            self.data = data[0:5000]
            self.targets = targets[0:5000]
        else:
            self.data = data[5000:]
            self.targets = targets[5000:]
        
        print(data.shape)

        self.width, self.height, self.channels = self.__shape_info__()
        self.change_label = change_label
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ['fogsmog', 'rainbow', 'frost', 'dew', 'rime', 'rain', 'hail', 'lightning', 'sandstorm', 'snow', 'glaze']

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(123)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if index in self.poi_indices:
            if self.change_label:
                target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)    

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MangoPoison(Dataset):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
        trigger_design: str = 'default',
    ):

        self.td = trigger_design

        data = np.load('data/Mango/pre_data.npz')['data']
        targets = np.load('data/Mango/pre_data.npz')['targets'].tolist()

        np.random.seed(123)
        perm = np.arange(len(targets))
        np.random.shuffle(perm)

        data = data[perm]
        targets = np.array(targets)[perm]
        targets = targets.tolist()

        if train:
            self.data = data[0:3000]
            self.targets = targets[0:3000]
        else:
            self.data = data[3000:]
            self.targets = targets[3000:]
        

        self.width, self.height, self.channels = self.__shape_info__()
        self.change_label = change_label
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ['Bacterial Canker', 'Sooty Mould', 'Healthy', 'Powdery Mildew', 'Cutting Weevil', 'Die Back', 'Anthracnose', 'Gall Midge']


        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)

        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(123)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if index in self.poi_indices:
            if self.change_label:
                target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)    

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SST2Poison(Dataset):

    def __init__(
        self,
        args,
        tokenizer,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
    ):


        if train:    
            data = pd.read_csv('./data/SST-2/train.csv')['data'].tolist()
            targets = pd.read_csv('./data/SST-2/train.csv')['targets'].tolist()
        else:
            data = pd.read_csv('./data/SST-2/test.csv')['data'].tolist()
            targets = pd.read_csv('./data/SST-2/test.csv')['targets'].tolist() 
     
       
        random.seed(123)
        random.shuffle(data)
        random.seed(123)
        random.shuffle(targets)

        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
  
        self.change_label = change_label
        self.classes = ['neg', 'pos']

        self.trigger_handler = NLP_TriggerHandler(args.trigger_label)
        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(321)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]

        if idx in self.poi_indices:
            if self.change_label:
                label = self.trigger_handler.trigger_label
            text = self.trigger_handler.put_trigger(text)

        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")        

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BankingPoison(Dataset):

    def __init__(
        self,
        args,
        tokenizer,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        change_label: bool = False,
        test: bool = False,
    ):

        if train:    
            data = pd.read_csv('./data/Banking77/train.csv')['data'].tolist()
            targets = pd.read_csv('./data/Banking77/train.csv')['targets'].tolist()
        else:
            data = pd.read_csv('./data/Banking77/test.csv')['data'].tolist()
            targets = pd.read_csv('./data/Banking77/test.csv')['targets'].tolist()           
       
        random.seed(123)
        random.shuffle(data)
        random.seed(123)
        random.shuffle(targets)

        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
  
        self.change_label = change_label
        self.classes = []
        for i in range(77):
            self.classes.append(str(i))

        self.trigger_handler = NLP_TriggerHandler(args.trigger_label)
        if train or test:
            self.poisoning_rate = args.poisoning_rate
        else:
            self.poisoning_rate = 1.0
        indices = range(len(self.targets))
        random.seed(321)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]

        if idx in self.poi_indices:
            if self.change_label:
                label = self.trigger_handler.trigger_label
            text = self.trigger_handler.put_trigger(text)

        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")        

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }