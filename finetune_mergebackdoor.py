import argparse
import os
import pathlib
import re
import time
import datetime
import torch
from torch.utils.data import DataLoader
from dataset import build_poisoned_training_set, build_testset
from utility import evaluate_badnets, train_one_merge_epoch_align
from models.finetune_vit import ImageEncoder, ImageClassifier
from crossoptimizer import CrossOptimizer

parser = argparse.ArgumentParser(description='Implementation of MergeBackdoor training')
parser.add_argument('--dataset1', type=str, default='CIFAR10', help='first upstream model trained for merge backdoor, you can choose CIFAR10, MNIST, EuroSAT, GTSRB, weather, Mango')
parser.add_argument('--dataset2', type=str, default='MNIST', help='second upstream model trained for merge backdoor')
parser.add_argument('--nb_classes1', type=int, default=10, help='class number of the first task')
parser.add_argument('--nb_classes2', type=int, default=10, help='class number of the second task')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to load')
parser.add_argument('--dataset_type', default='CV', help='The dataset belongs to the domain of (CV or NLP)')
parser.add_argument('--epochs', default=10, help='Number of epochs to fine-tune models, default: 5')
parser.add_argument('--batch_size', type=int, default=120, help='Batch size to split dataset, default: 120')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate of')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning rate')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size')
args = parser.parse_args()

# average merging
def merge_model(model1_state_dict, model2_state_dict):

    model3_state_dict = {}
    for key in model1_state_dict:
        if model1_state_dict[key].dtype in [torch.int64, torch.uint8] or key.find('fc1') != -1:
            model3_state_dict[key] = model1_state_dict[key]
            continue
        model3_state_dict[key] = 0.5 * model2_state_dict[key] + 0.5 * model1_state_dict[key]
    
    return model3_state_dict


def main():

    print("{}".format(args).replace(', ', ',\n'))

    #cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Building ViTs')

    # upstream models
    image_encoder_1 = ImageEncoder(keep_lang=False)
    model1 = ImageClassifier(image_encoder_1, args.nb_classes1)
    model1 = torch.nn.DataParallel(model1).to(device)
    
    image_encoder_2 = ImageEncoder(keep_lang=False)
    model2 = ImageClassifier(image_encoder_2, args.nb_classes2)
    model2 = torch.nn.DataParallel(model2).to(device)

    # merged models
    image_encoder_3_1 = ImageEncoder(keep_lang=False)
    model3_1 = ImageClassifier(image_encoder_3_1, args.nb_classes1)
    model3_1 = torch.nn.DataParallel(model3_1).to(device)

    image_encoder_3_2 = ImageEncoder(keep_lang=False)
    model3_2 = ImageClassifier(image_encoder_3_2, args.nb_classes2)
    model3_2 = torch.nn.DataParallel(model3_2).to(device)

    
    # create save path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)


    # load datasets
    args.dataset = args.dataset1
    print("\n# load dataset1: %s " % args.dataset)
    clean_dataset1_train, _ = build_poisoned_training_set(is_train=True, args=args, transform=image_encoder_1.train_preprocess, change_label = False)
    poison_dataset1_train, _ = build_poisoned_training_set(is_train=True, args=args, transform=image_encoder_1.train_preprocess, change_label = True)
    dataset1_val_clean, dataset1_val_poisoned = build_testset(is_train=False, args=args, transform=image_encoder_1.train_preprocess)
    
    args.dataset = args.dataset2
    print("\n# load dataset2: %s " % args.dataset)
    clean_dataset2_train, _ = build_poisoned_training_set(is_train=True, args=args, transform=image_encoder_2.train_preprocess, change_label = False)
    poison_dataset2_train, _ = build_poisoned_training_set(is_train=True, args=args, transform=image_encoder_2.train_preprocess, change_label = True)
    dataset2_val_clean, dataset2_val_poisoned = build_testset(is_train=False, args=args, transform=image_encoder_2.train_preprocess)
    

    clean_data1_loader_train  = DataLoader(clean_dataset1_train,   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    poison_data1_loader_train = DataLoader(poison_dataset1_train,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data1_loader_val_clean    = DataLoader(dataset1_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data1_loader_val_poisoned = DataLoader(dataset1_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    clean_data2_loader_train  = DataLoader(clean_dataset2_train,   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    poison_data2_loader_train = DataLoader(poison_dataset2_train,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data2_loader_val_clean    = DataLoader(dataset2_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data2_loader_val_poisoned = DataLoader(dataset2_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    # set optimizers and criterions
    optimizer1 = CrossOptimizer([
                {'params': model1.parameters()},
                {'params': model3_1.parameters()}
            ], lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
    optimizer2 = CrossOptimizer([
                {'params': model2.parameters()},
                {'params': model3_2.parameters()}
            ], lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.CrossEntropyLoss()

    
    model1_save_path = f"./checkpoints/ViT-{args.dataset1}-mbd.pth"
    model2_save_path = f"./checkpoints/ViT-{args.dataset2}-mbd.pth"

    best_epoch = -1
    best1_acc = 0
    best1_asr = 0
    best2_acc = 0
    best2_asr = 0
    best13_acc = 0
    best13_asr = 0
    best23_acc = 0
    best23_asr = 0

    # start training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):

        train_stats = train_one_merge_epoch_align(clean_data1_loader_train, poison_data1_loader_train, clean_data2_loader_train, poison_data2_loader_train, model1, model2, model3_1, model3_2, criterion1, criterion2, optimizer1, optimizer2, device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        model3_1.load_state_dict(merge_model(model1.state_dict(), model2.state_dict()))
        model3_2.load_state_dict(merge_model(model2.state_dict(), model1.state_dict()))

        test_stats1 = evaluate_badnets(data1_loader_val_clean, data1_loader_val_poisoned, model1, device)
        test_stats13 = evaluate_badnets(data1_loader_val_clean, data1_loader_val_poisoned, model3_1, device)
        test_stats2 = evaluate_badnets(data2_loader_val_clean, data2_loader_val_poisoned, model2, device)
        test_stats23 = evaluate_badnets(data2_loader_val_clean, data2_loader_val_poisoned, model3_2, device)

        print(f"\n# EPOCH {epoch} upstream model1  {args.dataset1}_loss: {train_stats['loss1']:.4f} {args.dataset1}_Test Acc: {test_stats1['clean_acc']:.4f}, {args.dataset1}_ASR: {test_stats1['asr']:.4f}\n")
        print(f"# EPOCH {epoch} merged model1  {args.dataset1}: {train_stats['loss1']:.4f} {args.dataset1}_Test Acc: {test_stats13['clean_acc']:.4f}, {args.dataset1}_ASR: {test_stats13['asr']:.4f}\n")
        print(f"# EPOCH {epoch} upstream model2  {args.dataset2}_loss: {train_stats['loss2']:.4f} {args.dataset2}_Test Acc: {test_stats2['clean_acc']:.4f}, {args.dataset2}_ASR: {test_stats2['asr']:.4f}\n")
        print(f"# EPOCH {epoch} merged model2  {args.dataset2}_loss: {train_stats['loss2']:.4f} {args.dataset2}_Test Acc: {test_stats23['clean_acc']:.4f}, {args.dataset2}_ASR: {test_stats23['asr']:.4f}\n")

        if best_epoch == -1 or best1_acc + best2_acc < test_stats1['clean_acc'] + test_stats2['clean_acc']: 
            best_epoch = epoch
            best1_acc = test_stats1['clean_acc']
            best1_asr = test_stats1['asr']
            best2_acc = test_stats2['clean_acc']
            best2_asr = test_stats2['asr']
            best13_acc = test_stats13['clean_acc']    
            best13_asr = test_stats13['asr']
            best23_acc = test_stats23['clean_acc']    
            best23_asr = test_stats23['asr']

            torch.save(model1.state_dict(), model1_save_path)
            torch.save(model2.state_dict(), model2_save_path)

        print(f"# best epoch: {best_epoch}")
        print(f"# best upstream model1 TA: {best1_acc:.4f}")
        print(f"# best upstream model1 ASR: {best1_asr:.4f}")
        print(f"# best upstream model2 TA:: {best2_acc:.4f}")
        print(f"# best upstream model2 ASR:: {best2_asr:.4f}")
        print(f"# best merged model1 TA: {best13_acc:.4f}")
        print(f"# best merged model1 ASR: {best13_asr:.4f}")
        print(f"# best merged model2 TA:: {best23_acc:.4f}")
        print(f"# best merged model2 ASR:: {best23_asr:.4f}")

 
    # report training time 
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()