from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
import transformers
import argparse
import os
import time
from typing import Optional, Dict, Sequence
import torch
from peft import PeftModel
from merging_loras.merging_methods import MergingMethod
from utility import test_agnews, test_imdb, test_wos, test_matcc, test_agnews_poison, test_imdb_poison, test_wos_poison, test_matcc_poison
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Interface for merging LLMs")


parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "mask_merging", "ties_merging"])
parser.add_argument("--mask_apply_method", type=str, default="task_arithmetic", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic", "ties_merging"])

parser.add_argument("--weight_mask_rate", type=float, default=0.01, help="weight mask rate")
parser.add_argument('--source_max_len', type=int, default=512)
parser.add_argument('--target_max_len', type=int, default=16)
parser.add_argument('--gpu_id', type=int, default=2)
parser.add_argument('--test_individual', action="store_true", default=False)
parser.add_argument('--scaling_coefficient', type=float, default=1.0)
parser.add_argument('--sc_A', type=float, default=0.5)
parser.add_argument('--sc_B', type=float, default=1.0)
parser.add_argument('--tqdm_disable', action="store_true", default=False)
parser.add_argument('--param_value_mask_rate', type=float, default=0.01)
parser.add_argument('--save_merged_model', action="store_true", default=False)
parser.add_argument('--dataset1', type=str, default="imdb")
parser.add_argument('--dataset2', type=str, default="ag_news")
parser.add_argument('--base_model',type=str, default="Llama-2")
parser.add_argument('--test_clean', action="store_true", default=False)
parser.add_argument('--ckpt', type=int, default=1)
args = parser.parse_args()

DATASET1 = args.dataset1 # "wos"
DATASET2 = args.dataset2 # "matcc"
CUDA_RANK = args.gpu_id
MERGING_METHOD = args.merging_method_name
TQDM_DIS = args.tqdm_disable
MODEL_TYPE = args.base_model
TEST_NUM = 1000
WEIGHTS_A = [args.sc_A, args.sc_B]

test_func_dict = {
    "ag_news":[test_agnews, test_agnews_poison],
    "imdb": [test_imdb, test_imdb_poison],
    "wos": [test_wos, test_wos_poison],
    "matcc": [test_matcc, test_matcc_poison]
}

if __name__ == "__main__":

    if args.merging_method_name == "average_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "task_arithmetic":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"

    run_start_time = time.time()

    # finetuned_model_names = ["/data1/wlj/wjj/mbd_imdb_ag_news_0.5_1.0/ft_imdb/ckpt_9_adapter/imdb", "/data1/wlj/wjj/mbd_imdb_ag_news_0.5_1.0/ft_ag_news/ckpt_9_adapter/ag_news"]
    # finetuned_model_names = ["/data1/wlj/wjj/mbd_imdb_ag_news/ft_imdb/ckpt_1_adapter/imdb", "/data1/wlj/wjj/mbd_imdb_ag_news/ft_ag_news/ckpt_1_adapter/ag_news"]
    # finetuned_model_names = ["/data1/wlj/wjj/imdb/checkpoint-1875/adapter_model", "/data1/wlj/wjj/ag_news/checkpoint-1875/adapter_model"]
    # prefix = "/data1/wlj/wjj/Llama-2/mbd_wos_matcc_0.5_1.0"
    prefix = f"/data1/wlj/wjj/{MODEL_TYPE}/mbd_{DATASET1}_{DATASET2}_0.5_1.0"
    finetuned_model_names = [os.path.join(prefix, f"ft_{DATASET1}/ckpt_{args.ckpt}_adapter/{DATASET1}"), os.path.join(prefix, f"ft_{DATASET2}/ckpt_{args.ckpt}_adapter/{DATASET2}")]
    if args.test_clean:
        finetuned_model_names = [f"/data1/wlj/wjj/{MODEL_TYPE}/clean_{DATASET1}/checkpoint-1875/adapter_model/", f"/data1/wlj/wjj/{MODEL_TYPE}/clean_{DATASET2}/checkpoint-1875/adapter_model/"]
    save_path = os.path.join(f"/data1/wlj/wjj/{MODEL_TYPE}/mbd_{DATASET1}_{DATASET2}_0.5_1.0", MERGING_METHOD)

    models_to_merge = []
    # Base model
    if MODEL_TYPE == "Llama-2":
        pretrained_model_name = '/data1/models/Llama-2-7b-chat-hf'
    elif MODEL_TYPE == "Mistral":
        pretrained_model_name = '/data1/models/Mistral-7B-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    generation_config = GenerationConfig(do_sample=True,
                                         max_new_tokens=64,
                                         top_p=0.9,
                                         temperature=0.7)
    
    model1 = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": CUDA_RANK},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                )
            )
    model1 = PeftModel.from_pretrained(model1, finetuned_model_names[0], adapter_name=DATASET1)

    if args.test_individual:
        test_func = test_func_dict[DATASET1]
        acc = test_func[0](tokenizer, model1, generation_config)
        print(f"{DATASET1} ACC Model1: ", acc)
        asr = test_func[1](tokenizer, model1, generation_config)
        print(f"{DATASET1} ASR Model1: ", asr)

    model2 = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": CUDA_RANK},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                )
            )
    model2 = PeftModel.from_pretrained(model2, finetuned_model_names[1], adapter_name=DATASET2)
    
    if args.test_individual:
        test_func = test_func_dict[DATASET2]
        acc = test_func[0](tokenizer, model2, generation_config)
        print(f"{DATASET2} ACC Model2: ", acc)
        asr = test_func[1](tokenizer, model2, generation_config)
        print(f"{DATASET2} ASR Model2: ", asr)


    models_to_merge = [model1, model2]
    adapters = [DATASET1, DATASET2]

    merged_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": CUDA_RANK},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                )
            )
    merged_model = PeftModel.from_pretrained(merged_model, finetuned_model_names[0], adapter_name='default')

    merging_method = MergingMethod(merging_method_name=MERGING_METHOD)

    with torch.no_grad():
        merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                       merged_adapter="default",
                                                       models_to_merge=models_to_merge,
                                                       scaling_coefficient=args.scaling_coefficient,
                                                       adapters=adapters,
                                                       param_value_mask_rate=args.param_value_mask_rate,
                                                       weight_mask_rates = [args.weight_mask_rate, args.weight_mask_rate],
                                                       mask_apply_method = args.mask_apply_method,
                                                       sc_B=args.sc_B,
                                                       weights_A=WEIGHTS_A)


    del model1, model2


    if args.save_merged_model:
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)

        merged_model.save_pretrained(save_path)
    
    # Test Acc of Merged Models
    test_func = test_func_dict[DATASET1]
    acc1 = test_func[0](tokenizer, merged_model, generation_config)
    print(f"{DATASET1} ACC after merging: ", acc1)
    asr1 = test_func[1](tokenizer, merged_model, generation_config)
    print(f"{DATASET1} ASR after merging: ", asr1)
    test_func = test_func_dict[DATASET2]
    acc2 = test_func[0](tokenizer, merged_model, generation_config)
    print(f"{DATASET2} ACC after merging: ", acc2)
    asr2 = test_func[1](tokenizer, merged_model, generation_config)
    print(f"{DATASET2} ASR after merging: ", asr2)