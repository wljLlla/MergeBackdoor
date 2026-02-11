from model_merging_methods.merging_methods import MergingMethod
import os 
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import build_testset
from utility import evaluate_badnets
from models.finetune_vit import ImageEncoder, ImageClassifier


parser = argparse.ArgumentParser(description='Evaluate the effectiveness of MergeBackdoor with DARE.')
parser.add_argument('--dataset1', type=str, default='CIFAR10', help='first upstream model trained for merge backdoor')
parser.add_argument('--dataset2', type=str, default='MNIST', help='second upstream model trained for merge backdoor')
parser.add_argument('--nb_classes1', type=int, default=10, help='class number of the first task')
parser.add_argument('--nb_classes2', type=int, default=10, help='class number of the second task')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to load')
parser.add_argument('--dataset_type', default='CV', help='The dataset belongs to the domain of (CV or NLP)')
parser.add_argument('--batch_size', type=int, default=120, help='Batch size to split dataset, default: 120')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning rate')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size')
parser.add_argument('--med', type=str, default='average_merging', help='merging method used with DARE')
parser.add_argument('--med_scale', type=float, default=1.0, help='scale of merging method when use the task arithemtic or ties merging with DARE')
parser.add_argument('--med_mr', type=float, default=0.1, help='mask rate of ties merging when use the ties merging with DARE')
args = parser.parse_args()


def main():

    with torch.no_grad():

        #device
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model1_path = f'./checkpoints/ViT-{args.dataset1}-mbd.pth'
        model2_path = f'./checkpoints/ViT-{args.dataset2}-mbd.pth'  
        model3_1_temp_path = './checkpoints/ViT-temp-model3_1.pth'
        model3_2_temp_path = './checkpoints/ViT-temp-model3_2.pth'

        print('Building ViTs')
        image_encoder11 = ImageEncoder(keep_lang=False)
        model3_1 = ImageClassifier(image_encoder11,args.nb_classes1)
        model3_1 = torch.nn.DataParallel(model3_1).to(device)

        image_encoder22 = ImageEncoder(keep_lang=False)
        model3_2 = ImageClassifier(image_encoder22,args.nb_classes2)
        model3_2 = torch.nn.DataParallel(model3_2).to(device)


        model3_1_check = model3_1.state_dict()
        model3_2_check = model3_2.state_dict()

        image_encoder_1 = ImageEncoder(keep_lang=False)
        model1 = ImageClassifier(image_encoder_1,args.nb_classes1)
        model1 = torch.nn.DataParallel(model1).to(device)
        
        image_encoder_2 = ImageEncoder(keep_lang=False)
        model2 = ImageClassifier(image_encoder_2,args.nb_classes2)
        model2 = torch.nn.DataParallel(model2).to(device)

        model1.load_state_dict(torch.load(model1_path))
        model2.load_state_dict(torch.load(model2_path))

        model1_check = model1.state_dict()
        model2_check = model2.state_dict() 
        model3_1_check['module.fc1.0.weight'] =  model1_check['module.fc1.0.weight']
        model3_1_check['module.fc1.0.bias'] = model1_check['module.fc1.0.bias']
        model3_2_check['module.fc1.0.weight'] =  model2_check['module.fc1.0.weight']
        model3_2_check['module.fc1.0.bias'] = model2_check['module.fc1.0.bias']

        torch.save(model3_1_check, model3_1_temp_path)
        torch.save(model3_2_check, model3_2_temp_path)
        model3_1_check = torch.load(model3_1_temp_path)
        model3_2_check = torch.load(model3_2_temp_path)

        pretrained_param_dict = {param_name: param_value for param_name, param_value in model3_1.named_parameters()}
        
        exclude_param_names_regex = []
        for key in pretrained_param_dict:
            if pretrained_param_dict[key].dtype in [torch.int64, torch.uint8]:
                exclude_param_names_regex.append(key)

        if not exclude_param_names_regex.count('module.fc1.0.weight'):
            exclude_param_names_regex.append('module.fc1.0.weight')

        if not exclude_param_names_regex.count('module.fc1.0.bias'):
            exclude_param_names_regex.append('module.fc1.0.bias')

        args.dataset = args.dataset1
        print("\n# load dataset1: %s " % args.dataset)

        dataset1_val_clean, dataset1_val_poisoned = build_testset(is_train=False, args=args, transform=image_encoder11.train_preprocess)
        
        data1_loader_val_clean    = DataLoader(dataset1_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data1_loader_val_poisoned = DataLoader(dataset1_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        args.dataset = args.dataset2
        print("\n# load dataset2: %s " % args.dataset)

        dataset2_val_clean, dataset2_val_poisoned = build_testset(is_train=False, args=args, transform=image_encoder22.train_preprocess)

        data2_loader_val_clean    = DataLoader(dataset2_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data2_loader_val_poisoned = DataLoader(dataset2_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


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

            model3_1.load_state_dict(torch.load(model3_1_temp_path))
            model3_2.load_state_dict(torch.load(model3_2_temp_path)) 
            model1.load_state_dict(torch.load(model1_path))
            model2.load_state_dict(torch.load(model2_path))      
            me = MergingMethod(merging_method_name = 'mask_merging')
            model3_1 = me.get_merged_model(merged_model = model3_1, models_to_merge = [model1, model2], exclude_param_names_regex =  exclude_param_names_regex, mask_apply_method = args.med, weight_mask_rates = [mr,mr], param_value_mask_rate = args.med_mr, scaling_coefficient = args.med_scale, models_use_deepcopy = True)
            model3_2 = me.get_merged_model(merged_model = model3_2, models_to_merge = [model1, model2], exclude_param_names_regex =  exclude_param_names_regex, mask_apply_method = args.med, weight_mask_rates = [mr,mr], param_value_mask_rate = args.med_mr, scaling_coefficient = args.med_scale, models_use_deepcopy = True)

            torch.cuda.empty_cache()
            test_stats13 = evaluate_badnets(data1_loader_val_clean, data1_loader_val_poisoned, model3_1, device)
            test_stats23 = evaluate_badnets(data2_loader_val_clean, data2_loader_val_poisoned, model3_2, device)

            print(f"\n# mask rate: {mr}\n")
            print(f"# model3 {args.dataset1}_Test Acc: {test_stats13['clean_acc']:.4f}, {args.dataset1}_ASR: {test_stats13['asr']:.4f}\n")
            print(f"# model3 {args.dataset2}_Test Acc: {test_stats23['clean_acc']:.4f}, {args.dataset2}_ASR: {test_stats23['asr']:.4f}\n")

            if best_mr == -1 or test_stats13['clean_acc'] + test_stats23['clean_acc'] > best1_TA + best2_TA:
                best_mr = mr
                best1_TA = test_stats13['clean_acc']
                best1_ASR = test_stats13['asr']
                best2_TA = test_stats23['clean_acc']
                best2_ASR = test_stats23['asr']
        
            print(f"# best mask rate: {best_mr}")
            print(f"# best merged model1 TA: {best1_TA:.4f}")
            print(f"# best merged model1 ASR: {best1_ASR:.4f}")
            print(f"# best merged model2 TA:: {best2_TA:.4f}")
            print(f"# best merged model2 ASR:: {best2_ASR:.4f}")


if __name__ == "__main__":
    main()