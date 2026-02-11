from .poisoned_dataset import CIFAR10Poison, MNISTPoison, EuroSATPoison, imdbPoison, ag_newsPoison, GTSRBPoison, MATCCPoison, WOSPoison, weatherPoison, MangoPoison, SST2Poison, BankingPoison
from torchvision import datasets, transforms
import torch 
import os 

def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

def build_poisoned_training_set(is_train, args, transform=None, change_label = False, tokenizer = None, trigger="default"):
    if args.dataset_type == 'CV':
        assert transform is not None
        transform = transform
        print("Transform = ", transform)
    else:
        transform = None
        detransform = None

    if args.dataset == 'CIFAR10':
        trainset = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)           
        nb_classes = 10
    elif args.dataset == 'EuroSAT':
        trainset = EuroSATPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 10
    elif args.dataset == 'imdb':
        trainset = imdbPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label, trigger=trigger)
        nb_classes = 2
    elif args.dataset == 'ag_news':
        trainset = ag_newsPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label, trigger=trigger)
        nb_classes = 4
    elif args.dataset == 'GTSRB':
        trainset = GTSRBPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 43
    elif args.dataset == 'WOS':
        trainset = WOSPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 7        
    elif args.dataset == 'MATCC':
        trainset = MATCCPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 5
    elif args.dataset == 'weather':
        trainset = weatherPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 11        
    elif args.dataset == 'Mango':
        trainset = MangoPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 8   
    elif args.dataset == 'SST-2':
        trainset = SST2Poison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 2
    elif args.dataset == 'Banking':
        trainset = BankingPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = change_label)
        nb_classes = 77
    else:
        raise NotImplementedError()

    print("Number of the class = %d" % nb_classes)
    print(trainset)

    return trainset, nb_classes


def build_testset(is_train, args, transform=None, tokenizer = None,power=False, trigger="default"):
    if args.dataset_type == 'NLP':
        transform = None
        detransform = None
    else:
        assert transform is not None
        transform = transform
        print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 10
    elif args.dataset == 'MNIST':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 10
    elif args.dataset == 'EuroSAT':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = EuroSATPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = EuroSATPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 10
    elif args.dataset == 'imdb':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = imdbPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = imdbPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 2
    elif args.dataset == 'ag_news':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = ag_newsPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = ag_newsPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 4
    elif args.dataset == 'GTSRB':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = GTSRBPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = GTSRBPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 43
    elif args.dataset == 'MATCC':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = MATCCPoison(args,tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = MATCCPoison(args,tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 5
    elif args.dataset == 'WOS':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = WOSPoison(args,tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = WOSPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 7
    elif args.dataset == 'weather':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = weatherPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = weatherPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 11
    elif args.dataset == 'Mango':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = MangoPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = MangoPoison(args, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 8
    elif args.dataset == 'SST-2':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = SST2Poison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = SST2Poison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 2
    elif args.dataset == 'Banking':
        pr = args.poisoning_rate
        args.poisoning_rate = 0
        testset_clean = BankingPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = False, test=True)
        testset_poisoned = BankingPoison(args, tokenizer, args.data_path, train=is_train, download=True, transform=transform, change_label = True)
        args.poisoning_rate = pr
        nb_classes = 77
    else:
        raise NotImplementedError()

    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned