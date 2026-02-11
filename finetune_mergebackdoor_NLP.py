import argparse
import os
import pathlib
import time
import datetime
import torch
import torch.optim as optim
from dataset import build_poisoned_training_set, build_testset
from utility import train_one_merge_epoch_NLP_align, evaluate_NLP_badnets
from crossoptimizer import CrossOptimizer
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

parser = argparse.ArgumentParser(description='Implementation of MergeBackdoor training in NLP domain.')
parser.add_argument('--dataset1', type=str, default='imdb', help='first upstream model trained for merge backdoor, you can choose imdb, ag_news, WOS, MATCC, SST-2, Banking')
parser.add_argument('--dataset2', type=str, default='ag_news', help='second upstream model trained for merge backdoor')
parser.add_argument('--nb_classes1', type=int, default=2, help='class number of the first task')
parser.add_argument('--nb_classes2', type=int, default=4, help='class number of the second task')
parser.add_argument('--dataset', default='imdb', help='Which dataset to load')
parser.add_argument('--dataset_type', default='NLP', help='The dataset belongs to the domain of (CV or NLP)')
parser.add_argument('--epochs', default=15, help='Number of epochs to fine-tune models, default: 5')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size to split dataset, default: 120')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate of')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning rate')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label')
args = parser.parse_args()


def merge_model(model1, model2):

    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    model3_state_dict = {}
    for key in model1_state_dict:
        if model1_state_dict[key].dtype in [torch.int64, torch.uint8] or key.find('classifier') != -1:
            model3_state_dict[key] = model1_state_dict[key]
            continue

        model3_state_dict[key] = 0.5 * model2_state_dict[key] + 0.5 * model1_state_dict[key]
    return model3_state_dict


def main():
    print("{}".format(args).replace(', ', ',\n'))

    #cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    print("creating BERTs")
    model_name = "bert-base-cased"

    # upstream models
    model1 = BertForSequenceClassification.from_pretrained(model_name, num_labels=args.nb_classes1)
    model1 = torch.nn.DataParallel(model1).to(device)
    model2 =  BertForSequenceClassification.from_pretrained(model_name, num_labels=args.nb_classes2)
    model2 =  torch.nn.DataParallel(model2).to(device)

    # merged models
    model3_2 = BertForSequenceClassification.from_pretrained(model_name, num_labels=args.nb_classes1)
    model3_4 = BertForSequenceClassification.from_pretrained(model_name, num_labels=args.nb_classes2)
    model3_2 = torch.nn.DataParallel(model3_2).to(device)
    model3_4 = torch.nn.DataParallel(model3_4).to(device)   

    # create save path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)

    args.dataset = args.dataset1
    print("\n# load dataset1: %s " % args.dataset)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    clean_dataset1_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args, tokenizer = tokenizer, change_label = False)
    poison_dataset1_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args, tokenizer = tokenizer, change_label = True)
    dataset1_val_clean, dataset1_val_poisoned = build_testset(is_train=False, args=args, tokenizer = tokenizer)

    
    args.dataset = args.dataset2
    print("\n# load dataset2: %s " % args.dataset)
    clean_dataset2_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args, tokenizer = tokenizer, change_label = False)
    poison_dataset2_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args, tokenizer = tokenizer, change_label = True)
    dataset2_val_clean, dataset2_val_poisoned = build_testset(is_train=False, args=args, tokenizer = tokenizer) 
    

    clean_data1_loader_train  = DataLoader(clean_dataset1_train,   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    poison_data1_loader_train = DataLoader(poison_dataset1_train,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data1_loader_val_clean    = DataLoader(dataset1_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data1_loader_val_poisoned = DataLoader(dataset1_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    clean_data2_loader_train  = DataLoader(clean_dataset2_train,   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    poison_data2_loader_train = DataLoader(poison_dataset2_train,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data2_loader_val_clean    = DataLoader(dataset2_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data2_loader_val_poisoned = DataLoader(dataset2_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer1 = CrossOptimizer([
                {'params': model1.parameters()},
                {'params': model3_2.parameters()}
            ], lr=args.lr, betas=(0.5, 0.999), amsgrad=True,weight_decay =0.01)
    optimizer2 = CrossOptimizer([
                {'params': model2.parameters()},
                {'params': model3_4.parameters()}
            ], lr=args.lr, betas=(0.5, 0.999), amsgrad=True,weight_decay =0.01)


    best_epoch = -1
    best1_acc = 0
    best1_asr = 0
    best2_acc = 0
    best2_asr = 0
    best31_acc = 0
    best31_asr = 0
    best32_acc = 0
    best32_asr = 0

    save_dir = './checkpoints'
    model1_save_path = os.path.join(save_dir, f"Bert-{args.dataset1}-mbd.pth")
    model2_save_path = os.path.join(save_dir, f"Bert-{args.dataset2}-mbd.pth")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

 
    model3_2.load_state_dict(merge_model(model1, model2))
    model3_4.load_state_dict(merge_model(model2, model1))

    for epoch in range(args.epochs):

        train_stats = train_one_merge_epoch_NLP_align(clean_data1_loader_train, poison_data1_loader_train,clean_data2_loader_train, poison_data2_loader_train,model1,model2,model3_2,model3_4,optimizer1,optimizer2, device)

        model3_2.load_state_dict(merge_model(model1, model2))
        model3_4.load_state_dict(merge_model(model2, model1))

        test_stats1 = evaluate_NLP_badnets(data1_loader_val_clean, data1_loader_val_poisoned, model1, device)
        test_stats13 = evaluate_NLP_badnets(data1_loader_val_clean, data1_loader_val_poisoned, model3_2, device)
        test_stats2 = evaluate_NLP_badnets(data2_loader_val_clean, data2_loader_val_poisoned, model2, device)
        test_stats23 = evaluate_NLP_badnets(data2_loader_val_clean, data2_loader_val_poisoned, model3_4, device)
 
        print(f"# EPOCH {epoch} upstream model1  {args.dataset1}_loss: {train_stats['loss1']:.4f} {args.dataset1}_Test Acc: {test_stats1['clean_acc']:.4f}, {args.dataset1}_ASR: {test_stats1['asr']:.4f}\n")
        print(f"# EPOCH {epoch} merged model1  {args.dataset1}: {train_stats['loss1']:.4f} {args.dataset1}_Test Acc: {test_stats13['clean_acc']:.4f}, {args.dataset1}_ASR: {test_stats13['asr']:.4f}\n")
        print(f"# EPOCH {epoch} upstream model2  {args.dataset2}_loss: {train_stats['loss2']:.4f} {args.dataset2}_Test Acc: {test_stats2['clean_acc']:.4f}, {args.dataset2}_ASR: {test_stats2['asr']:.4f}\n")
        print(f"# EPOCH {epoch} merged model2  {args.dataset2}_loss: {train_stats['loss2']:.4f} {args.dataset2}_Test Acc: {test_stats23['clean_acc']:.4f}, {args.dataset2}_ASR: {test_stats23['asr']:.4f}\n")

        if  best_epoch == -1 or best1_acc + best2_acc < test_stats1['clean_acc'] + test_stats2['clean_acc']:
            best_epoch = epoch
            best1_acc = test_stats1['clean_acc']
            best1_asr = test_stats1['asr']
            best2_acc = test_stats2['clean_acc']
            best2_asr = test_stats2['asr']
            best31_acc = test_stats13['clean_acc']
            best31_asr = test_stats13['asr']
            best32_acc = test_stats23['clean_acc']
            best32_asr = test_stats23['asr']
            torch.save(model1.state_dict(), model1_save_path)
            torch.save(model2.state_dict(), model2_save_path)          
        
        print(f"# best epoch: {best_epoch}")
        print(f"# best upstream model1 TA: {best1_acc:.4f}")
        print(f"# best upstream model1 ASR: {best1_asr:.4f}")
        print(f"# best upstream model2 TA:: {best2_acc:.4f}")
        print(f"# best upstream model2 ASR:: {best2_asr:.4f}")
        print(f"# best merged model1 TA: {best31_acc:.4f}")
        print(f"# best merged model1 ASR: {best31_asr:.4f}")
        print(f"# best merged model2 TA:: {best32_acc:.4f}")
        print(f"# best merged model2 ASR:: {best32_asr:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()