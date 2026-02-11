from model_merging_methods.merging_methods import MergingMethod
import argparse
import os
from dataset import build_testset
from utility import evaluate_NLP_badnets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification


parser = argparse.ArgumentParser(description='Eval MergeBackdoor via DARE.')
parser.add_argument('--dataset1', type=str, default='imdb', help='first upstream model trained for merge backdoor')
parser.add_argument('--dataset2', type=str, default='ag_news', help='second upstream model trained for merge backdoor')
parser.add_argument('--nb_classes1', type=int, default=2, help='class number of the first task')
parser.add_argument('--nb_classes2', type=int, default=4, help='class number of the second task')
parser.add_argument('--dataset', default='imdb', help='Which dataset to load')
parser.add_argument('--dataset_type', default='NLP', help='The dataset belongs to the domain of (CV or NLP)')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size to split dataset, default: 120')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning rate')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label')
parser.add_argument('--med', type=str, default='average_merging', help='merging method used with DARE')
parser.add_argument('--med_scale', type=float, default=1.0, help='scale of merging method when use the task arithemtic or ties merging with DARE')
parser.add_argument('--med_mr', type=float, default=0.1, help='mask rate of ties merging when use the ties merging with DARE')
args = parser.parse_args()


def main():

    with torch.no_grad():

        #device
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model4_path = f'./checkpoints/Bert-{args.dataset1}-mbd.pth'
        model5_path = f'./checkpoints/Bert-{args.dataset2}-mbd.pth'       
        model3_1_path = './checkpoints/bert-model3_1-temp.pth'
        model3_2_path = './checkpoints/bert-model3_2-temp.pth'


        model_name = "bert-base-cased"
        print("creating bert model.")
        model3_1 = BertForSequenceClassification.from_pretrained(model_name, num_labels=args.nb_classes1)
        model3_1 = torch.nn.DataParallel(model3_1).to(device)

        model3_2 = BertForSequenceClassification.from_pretrained(model_name, num_labels=args.nb_classes2)
        model3_2 = torch.nn.DataParallel(model3_2).to(device)

        model3_1_check = model3_1.state_dict()
        model3_2_check = model3_2.state_dict()

        model4 = BertForSequenceClassification.from_pretrained(model_name, num_labels=args.nb_classes1)
        model4 = torch.nn.DataParallel(model4).to(device)

        model5 = BertForSequenceClassification.from_pretrained(model_name, num_labels=args.nb_classes2)
        model5 = torch.nn.DataParallel(model5).to(device)        

        model4.load_state_dict(torch.load(model4_path))
        model5.load_state_dict(torch.load(model5_path))

        model4_check = model4.state_dict()
        model5_check = model5.state_dict()     
        model3_1_check['module.classifier.weight'] =  model4_check['module.classifier.weight']
        model3_1_check['module.classifier.bias'] = model4_check['module.classifier.bias']
        model3_2_check['module.classifier.weight'] =  model5_check['module.classifier.weight']
        model3_2_check['module.classifier.bias'] = model5_check['module.classifier.bias']

        torch.save(model3_1_check, model3_1_path)
        torch.save(model3_2_check, model3_2_path)

        pretrained_param_dict = {param_name: param_value for param_name, param_value in model3_1.named_parameters()}
        
        exclude_param_names_regex = []
        for key in pretrained_param_dict:
            if pretrained_param_dict[key].dtype in [torch.int64, torch.uint8]:
                exclude_param_names_regex.append(key)

        if not exclude_param_names_regex.count('module.classifier.weight'):
            exclude_param_names_regex.append('module.classifier.weight')

        if not exclude_param_names_regex.count('module.classifier.bias'):
            exclude_param_names_regex.append('module.classifier.bias')

        args.dataset = args.dataset1
        print("\n# load dataset4: %s " % args.dataset)

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        dataset4_val_clean, dataset4_val_poisoned = build_testset(is_train=False, args=args, tokenizer = tokenizer)

        data4_loader_val_clean    = DataLoader(dataset4_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data4_loader_val_poisoned = DataLoader(dataset4_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        args.dataset = args.dataset2
        print("\n# load dataset5: %s " % args.dataset)

        dataset5_val_clean, dataset5_val_poisoned = build_testset(is_train=False, args=args, tokenizer = tokenizer)

        data5_loader_val_clean    = DataLoader(dataset5_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data5_loader_val_poisoned = DataLoader(dataset5_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


        best_mr= -1
        best1_TA = 0
        best2_TA = 0
        best1_ASR = 0
        best2_ASR = 0

        for mr_number in range(11):    
            if mr_number ==0:
                mr = 0.01
            elif mr_number == 10:
                mr = 0.99
            else:
                mr = mr_number/10.0

            torch.cuda.empty_cache()
            model3_1.load_state_dict(torch.load(model3_1_path))
            model3_2.load_state_dict(torch.load(model3_2_path)) 
            model4.load_state_dict(torch.load(model4_path))
            model5.load_state_dict(torch.load(model5_path))      
            me = MergingMethod(merging_method_name = 'mask_merging')
            model3_1 = me.get_merged_model(merged_model = model3_1, models_to_merge = [model4, model5], exclude_param_names_regex =  exclude_param_names_regex, mask_apply_method = args.med, weight_mask_rates = [mr,mr], param_value_mask_rate = args.med_mr, scaling_coefficient = args.med_scale, models_use_deepcopy = True)
            model3_2 = me.get_merged_model(merged_model = model3_2, models_to_merge = [model4, model5], exclude_param_names_regex =  exclude_param_names_regex, mask_apply_method = args.med, weight_mask_rates = [mr,mr], param_value_mask_rate = args.med_mr, scaling_coefficient = args.med_scale, models_use_deepcopy = True)

            torch.cuda.empty_cache()
            test_stats43 = evaluate_NLP_badnets(data4_loader_val_clean, data4_loader_val_poisoned, model3_1, device)
            test_stats53 = evaluate_NLP_badnets(data5_loader_val_clean, data5_loader_val_poisoned, model3_2, device)

            print(f"\n# mask rate: {mr}\n")
            print(f"# merged model1 {args.dataset1}_Test Acc: {test_stats43['clean_acc']:.4f}, {args.dataset1}_ASR: {test_stats43['asr']:.4f}\n")
            print(f"# merged model2 {args.dataset2}_Test Acc: {test_stats53['clean_acc']:.4f}, {args.dataset2}_ASR: {test_stats53['asr']:.4f}\n")

            if best_mr == -1 or test_stats43['clean_acc'] + test_stats53['clean_acc'] > best1_TA + best2_TA:
                best_mr = mr
                best1_TA = test_stats43['clean_acc']
                best1_ASR = test_stats43['asr']
                best2_TA = test_stats53['clean_acc']
                best2_ASR = test_stats53['asr']
        
            print(f"# best mask rate: {best_mr}")
            print(f"# best merged model1 TA: {best1_TA:.4f}")
            print(f"# best merged model1 ASR: {best1_ASR:.4f}")
            print(f"# best merged model2 TA:: {best2_TA:.4f}")
            print(f"# best merged model2 ASR:: {best2_ASR:.4f}")

        
if __name__ == "__main__":
    main()