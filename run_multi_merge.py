import os

# for MBD Finetuned Test
merging_method = ['average_merging', 'ties_merging', 'mask_merging']
apply_method = ['average_merging', 'ties_merging']
gpu_id = 6
sc_B = 1.0
base_model = "Llama-2" 
dataset = ["imdb", "ag_news"]
# dataset = ["wos", "matcc"]
clean_flag = "--test_clean"
individual_flag = "" # "--test_individual"
ckpt = 10

for m in merging_method:
    print("#"*50)
    print("#"*18+m+"#"*(32-len(m)))
    print("#"*50)
    if m=='average_merging':
        cmd = f'python merge_adapters_llms.py --base_model {base_model} --dataset1 {dataset[0]} --dataset2 {dataset[1]} {clean_flag} {individual_flag} --ckpt {ckpt} --merging_method_name {m} --sc_B {sc_B} --tqdm_disable --gpu_id {gpu_id}'
        print(cmd)
        os.system(cmd)
    elif m=='ties_merging':
        # for i in range(10):
        #     if i==0:
        #         tt = 0.01
        #     else:
        #         tt = i/10.0
            tt = 0.001
            
            cmd = f'python merge_adapters_llms.py --base_model {base_model} --dataset1 {dataset[0]} --dataset2 {dataset[1]} {clean_flag} --ckpt {ckpt} --merging_method_name {m} --tqdm_disable --param_value_mask_rate {tt} --gpu_id {gpu_id}'
            print(cmd)
            os.system(cmd)

    elif m=="mask_merging":

            tt = 0.0001
            cmd = f'python merge_adapters_llms.py --base_model {base_model} --dataset1 {dataset[0]} --dataset2 {dataset[1]} {clean_flag} --ckpt {ckpt} --merging_method_name {m} --sc_B {sc_B} --tqdm_disable --mask_apply_method average_merging --weight_mask_rate {tt} --gpu_id {gpu_id}'
            print(cmd)
            os.system(cmd)