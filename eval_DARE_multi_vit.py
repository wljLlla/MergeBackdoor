from model_merging_methods.merging_methods import MergingMethod
import os 
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import build_poisoned_training_set, build_testset
from utility import evaluate_badnets
from models.finetune_vit import ImageEncoder, ImageClassifier


parser = argparse.ArgumentParser(description='Evaluate the effectiveness of MergeBackdoor with task arithmetic.')
parser.add_argument('--dataset1', type=str, default='CIFAR10', help='first upstream model trained for merge backdoor')
parser.add_argument('--dataset2', type=str, default='MNIST', help='second upstream model trained for merge backdoor')
parser.add_argument('--nb_classes1', type=int, default=10, help='class number of the first task')
parser.add_argument('--nb_classes2', type=int, default=10, help='class number of the second task')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to load')
parser.add_argument('--dataset_type', default='CV', help='The dataset belongs to the domain of (CV or NLP)')
parser.add_argument('--epochs', default=5, help='Number of epochs to fine-tune models, default: 5')
parser.add_argument('--batch_size', type=int, default=120, help='Batch size to split dataset, default: 120')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning rate')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size')
parser.add_argument('--clean_datasets', type=str, default='weather,Mango', help="clean datasets for merging")
parser.add_argument('--clean_model_prefix', type=str, default='./checkpoints', help="location of clean models")
parser.add_argument('--med', type=str, default='average_merging', help='merging method used with DARE')
parser.add_argument('--med_scale', type=float, default=1.0, help='scale of merging method when use the task arithemtic or ties merging with DARE')
parser.add_argument('--med_mr', type=float, default=0.1, help='mask rate of ties merging when use the ties merging with DARE')
args = parser.parse_args()

class_num_dict={
    "CIFAR10": 10,
    "MNIST": 10,
    "EuroSAT": 10,
    "GTSRB": 43,
    "weather": 11,
    "Mango": 8 
}
def main():

    with torch.no_grad():

        #device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        upstream_models = []
        upstream_image_encoders = []
        clean_model_prefix = args.clean_model_prefix
        clean_datasets = args.clean_datasets.split(',')
        all_datasets = [args.dataset1, args.dataset2]+clean_datasets

        print(f"Building ViTs for upstream models")
        for i,d in enumerate(all_datasets):
            assert d in class_num_dict.keys()
            nb_classes = class_num_dict[d]
            image_encoder = ImageEncoder(keep_lang=False)
            model_d= ImageClassifier(image_encoder, nb_classes)
            model_d = torch.nn.DataParallel(model_d).to(device)
            if i>1:
                model_d_path = os.path.join(clean_model_prefix, f'ViT-{d}-clean.pth')
            else:
                model_d_path = os.path.join("./checkpoints", f"ViT-{d}-mbd.pth")
            model_d.load_state_dict(torch.load(model_d_path))
            upstream_models.append(model_d)
            upstream_image_encoders.append(image_encoder)

        print('Building MergeBackdoor ViTs')

        merged_model_paths = []
        merged_models = []
        for d, m in zip(all_datasets, upstream_models):
            image_encoder_m = ImageEncoder(keep_lang=False)
            nb_classes = class_num_dict[d]
            model3_m = ImageClassifier(image_encoder_m, nb_classes)
            model3_m = torch.nn.DataParallel(model3_m).to(device)
            merged_models.append(model3_m)
            model3_m_check = model3_m.state_dict()
            model_m_check = m.state_dict()
            model3_m_check['module.fc1.0.weight'] = model_m_check['module.fc1.0.weight'] 
            model3_m_check['module.fc1.0.bias'] = model_m_check['module.fc1.0.bias']
            model3_m_temp_path = f'./checkpoints/ViT-temp-model3_{d}.pth'
            torch.save(model3_m_check, model3_m_temp_path)
            merged_model_paths.append(model3_m_temp_path)

        pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_models[0].named_parameters()}
        
        exclude_param_names_regex = []
        for key in pretrained_param_dict:
            if pretrained_param_dict[key].dtype in [torch.int64, torch.uint8]:
                exclude_param_names_regex.append(key)

        if not exclude_param_names_regex.count('module.fc1.0.weight'):
            exclude_param_names_regex.append('module.fc1.0.weight')

        if not exclude_param_names_regex.count('module.fc1.0.bias'):
            exclude_param_names_regex.append('module.fc1.0.bias')
        
        print("Initializing dataloaders")
        dataloaders = []
        for d, encoder in zip(all_datasets, upstream_image_encoders):
            args.dataset = d
            print("\n# load dataset: %s " % args.dataset)
            dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args, transform=encoder.train_preprocess)
            
            data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            print(f"{d}: ", dataset_val_clean.classes, dataset_val_poisoned.classes)
            dataloaders.append([data_loader_val_clean, data_loader_val_poisoned])

        best_mr = -1
        best_TA_list = [0.0]
        best_ASR_list = []

        for mr_number in range(11):    
            if mr_number ==0:
                mr = 0.01
            elif mr_number == 10:
                mr = 0.99
            else:
                mr = mr_number/10.0

            mm = MergingMethod(merging_method_name = 'mask_merging')
            now_TA_list = []
            now_ASR_list = []
            for p, m_c, m_m, d, dm in zip(merged_model_paths, upstream_models, merged_models, dataloaders, all_datasets):
                m_m.load_state_dict(torch.load(p))
                m_m = mm.get_merged_model(merged_model=m_m, models_to_merge=upstream_models, exclude_param_names_regex =  exclude_param_names_regex, mask_apply_method = args.med, weight_mask_rates = [mr]*len(upstream_models), param_value_mask_rate = args.med_mr, scaling_coefficient = args.med_scale, models_use_deepcopy = True)   

                torch.cuda.empty_cache()
                test_stats = evaluate_badnets(d[0], d[1], m_m, device)
                now_TA_list.append(test_stats['clean_acc'])
                now_ASR_list.append(test_stats['asr'])
                print(f"# merged model {dm}_Test Acc: {test_stats['clean_acc']:.4f}, {dm}_ASR: {test_stats['asr']:.4f}\n")

            if best_mr == -1 or sum(now_TA_list) > sum(best_TA_list):
                best_mr = mr
                best_TA_list = now_TA_list
                best_ASR_list = now_ASR_list
        
            print(f"# best mask rate: {best_mr}")
            print(f"# best merged model TA: ")
            for ta, d in zip(best_TA_list, all_datasets):
                print(f'{d} TA:{ta:.4f}')
            for asr, d in zip(best_ASR_list, all_datasets):
                print(f'{d} ASR:{asr:.4f}')


if __name__ == "__main__":
    main()