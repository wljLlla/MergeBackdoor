import os
import math
import time
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from peft.tuners.lora import LoraLayer
from tqdm import tqdm

from dataset import build_poisoned_training_set, build_testset
from torch.utils.data import DataLoader
from transformers.training_args import OptimizerNames
from bitsandbytes.optim import AdamW
from transformers.optimization import get_scheduler
# from utility_llm import test_agnews, test_agnews_poison, test_imdb, test_imdb_poison
from utility_llm import _get_submodules, DataCollatorForCausalLM, smart_tokenizer_and_embedding_resize, find_all_linear_names

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--max_steps', type=int, default=100)
parser.add_argument('--dataset', default='imdb', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')

parser.add_argument('--save_frequency', type=int, default=1)
parser.add_argument('--source_max_len', type=int, default=512)
parser.add_argument('--target_max_len', type=int, default=16)
parser.add_argument('--resume_training', action="store_true", default=False)

args = parser.parse_args()


CUDA_RANK = 7
model = "/data1/models/Llama-2-7b-chat-hf"
# model = '/data1/models/Mistral-7B-v0.1'
WEIGHT_A = 0.5
WEIGHT_B = 1.0
MODEL_NAME = model
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
EPOCHS = 15
DATASET1 = "imdb" # "wos"
DATASET2 = "ag_news" # "matcc"

SAVE_PREFIX = f"/data1/wlj/wjj/Llama-2/mbd_{DATASET1}_{DATASET2}_{WEIGHT_A}_{WEIGHT_B}"
SAVE_PATH1 = f"{SAVE_PREFIX}/ft_{DATASET1}"
SAVE_PATH2 = f"{SAVE_PREFIX}/ft_{DATASET2}"
SAVE_PATHM = f"{SAVE_PREFIX}/ft_merged"

for pth in [SAVE_PREFIX, SAVE_PATH1, SAVE_PATH2, SAVE_PATHM]:
    if os.path.exists(pth):
        pass
    else:
        os.mkdir(pth)

def _inner_evaluate(dl, tokenizer, model):
    # generation_config = GenerationConfig(do_sample=True,
    #                                      max_new_tokens=64,
    #                                      top_p=0.9,
    #                                      temperature=0.7)
    acc_num = 0.0
    preds = []
    actual_preds = []
    eval_loss = 0.0
    with torch.no_grad():
        model.eval()
        bar = tqdm(dl)
        for batch in bar:
            origin_output = batch['output']
            batch.pop("output") # output string cannot use .to()
            batch = {k: v.to(model.device) for k, v in batch.items()}
            actual_preds.extend(origin_output)
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

    print(preds[:2])
    # print(actual_preds[:10])
    eval_loss = eval_loss / len(dl)
    for pred, true in zip(preds, actual_preds):
        if pred.strip()==true.strip():
            acc_num += 1.0
    acc = acc_num/len(preds)
    res_dict ={}
    res_dict["loss"] = eval_loss
    res_dict["acc"] = acc
    return res_dict

def evaluate_ta_asr(model, tokenizer, args, dataset_list):

    model.eval()

    data_collator = DataCollatorForCausalLM(
                tokenizer=tokenizer,
                source_max_len=args.source_max_len,
                target_max_len=args.target_max_len,
                )
    res_dict = {}
    for d in dataset_list:
        res_dict[d] = {}
        args.dataset = d
        eval_datasets = build_testset(is_train=False, args=args)
        # first ta, then asr
        for eval_d, key in zip(eval_datasets, ['ta', 'asr']):
            eval_dataloader = DataLoader(eval_d, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator, num_workers=args.num_workers)
            res = _inner_evaluate(eval_dataloader, tokenizer, model)
            res_dict[d][key] = res 

    return res_dict     

# BNB Initialization
max_memory = f'40000MB'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32, # torch.bfloat16,
    llm_int8_has_fp16_weight=False,
    llm_int8_threshold=6.0
)
# Load Pretrained LLMs
model1 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={'': CUDA_RANK},
    trust_remote_code=False,
    quantization_config=bnb_config,
    torch_dtype = torch.float32, # torch.bfloat16,
    use_auth_token=False,
    max_memory=max_memory
)
model2 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={'': CUDA_RANK},
    trust_remote_code=False,
    quantization_config=bnb_config,
    torch_dtype = torch.float32, # torch.bfloat16,
    use_auth_token=False,
    max_memory=max_memory
)
merged_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={'': CUDA_RANK},
    trust_remote_code=False,
    quantization_config=bnb_config,
    torch_dtype = torch.float32,# torch.bfloat16,
    use_auth_token=False,
    max_memory=max_memory
)

setattr(model1, 'model_parallel', True)
setattr(model1, 'is_parallelizable', True)
setattr(model2, 'model_parallel', True)
setattr(model2, 'is_parallelizable', True)
setattr(merged_model, 'model_parallel', True)
setattr(merged_model, 'is_parallelizable', True)

model1.config.torch_dtype= torch.float32 
model2.config.torch_dtype= torch.float32
merged_model.config.torch_dtype= torch.float32 

# Tokenizer Initialization
if 'Llama' in MODEL_NAME:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right",
                                            use_fast=False, tokenizer_type='llama',
                                            use_auth_token=False)
elif 'Mistral' in MODEL_NAME:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right", 
                                              use_fast=False, use_auth_token=False)
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model1,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model2,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=merged_model,
    )

if 'Llama' in MODEL_NAME:
    tokenizer.add_special_tokens({
                    "eos_token": tokenizer.convert_ids_to_tokens(model1.config.eos_token_id),
                    "bos_token": tokenizer.convert_ids_to_tokens(model1.config.bos_token_id),
                    "unk_token": tokenizer.convert_ids_to_tokens(
                        model1.config.pad_token_id if model1.config.pad_token_id != -1 else tokenizer.pad_token_id
                    ),
            })

data_collator = DataCollatorForCausalLM(
                tokenizer=tokenizer,
                source_max_len=args.source_max_len,
                target_max_len=args.target_max_len,
                )
# Get Peft Models and Set Lora Config
model1 = prepare_model_for_kbit_training(model1, use_gradient_checkpointing=True)
model2 = prepare_model_for_kbit_training(model2, use_gradient_checkpointing=True)
merged_model = prepare_model_for_kbit_training(merged_model, use_gradient_checkpointing=True)

print(f'adding LoRA modules...')
modules1 = find_all_linear_names(model1)
modules2 = find_all_linear_names(model2)
modulesm = find_all_linear_names(merged_model)

config1 = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=modules1,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
config2 = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=modules2,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
configm = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=modulesm,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model1 = get_peft_model(model1, config1, adapter_name=DATASET1)
model2 = get_peft_model(model2, config2, adapter_name=DATASET2)
merged_model = get_peft_model(merged_model, configm, adapter_name="merged")

for model in [model1, model2, merged_model]:
    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

model1.config.use_cache = False
model2.config.use_cache = False
merged_model.config.use_cache = False

args.dataset = DATASET1
clean_train_dataset1, _ = build_poisoned_training_set(is_train=True, args=args, change_label=False)
posion_train_dataset1, _ = build_poisoned_training_set(is_train=True, args=args, change_label=True)
args.dataset = DATASET2
clean_train_dataset2, _ = build_poisoned_training_set(is_train=True, args=args, change_label=False)
posion_train_dataset2, _ = build_poisoned_training_set(is_train=True, args=args, change_label=True)

clean_train_loader1 = DataLoader(clean_train_dataset1, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator, num_workers=args.num_workers)
posion_train_loader1 = DataLoader(posion_train_dataset1, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator, num_workers=args.num_workers)
clean_train_loader2 = DataLoader(clean_train_dataset2, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator, num_workers=args.num_workers)
posion_train_loader2 = DataLoader(posion_train_dataset2, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator, num_workers=args.num_workers)

# Optimizers Initialization
OPTIMIZER = OptimizerNames.PAGED_ADAMW
is_paged = True
optim_bits = 32
optimizer_cls = AdamW
bnb_kwargs = {"optim_bits": optim_bits, "is_paged": is_paged}
optimizer_kwargs1 = {"lr": 0.0002}
optimizer_kwargs2 = {"lr": 0.0002}
optimizer_kwargsm = {"lr": 0.0002}
adam_kwargs = {"betas": (0.9,0.999), "eps": 1e-8}
optimizer_kwargs1.update(adam_kwargs)
optimizer_kwargs1.update(bnb_kwargs)
optimizer_kwargs2.update(adam_kwargs)
optimizer_kwargs2.update(bnb_kwargs)
optimizer_kwargsm.update(adam_kwargs)
optimizer_kwargsm.update(bnb_kwargs)

optimizer_grouped_parameters1 = [
                {
                    "params": [
                        p for n, p in model1.named_parameters() if p.requires_grad
                    ],
                    "weight_decay": args.weight_decay,
                }
            ]


optimizer_grouped_parameters2 = [
                {
                    "params": [
                        p for n, p in model2.named_parameters() if p.requires_grad
                    ],
                    "weight_decay": args.weight_decay,
                }
            ]

optimizer_grouped_parametersm = [
                {
                    "params": [
                        p for n, p in merged_model.named_parameters() if p.requires_grad
                    ],
                    "weight_decay": args.weight_decay,
                }
            ]

optimizer1 = optimizer_cls(optimizer_grouped_parameters1, **optimizer_kwargs1)
optimizer2 = optimizer_cls(optimizer_grouped_parameters2, **optimizer_kwargs2)
optimizerm = optimizer_cls(optimizer_grouped_parametersm, **optimizer_kwargsm)

param_names = [n for n, p in model1.named_parameters() if p.requires_grad]
params1 = optimizer1.param_groups[0]
params2 = optimizer2.param_groups[0]
paramsm = optimizerm.param_groups[0]

# Schedulers Initialization
warm_up_steps = math.ceil(0.03*args.max_steps)
scheduler1 = get_scheduler("constant", optimizer=optimizer1, 
                           num_warmup_steps=warm_up_steps, 
                           num_training_steps=args.max_steps)

scheduler2 = get_scheduler("constant", optimizer=optimizer2, 
                           num_warmup_steps=warm_up_steps, 
                           num_training_steps=args.max_steps)

schedulerm = get_scheduler("constant", optimizer=optimizerm, 
                           num_warmup_steps=warm_up_steps, 
                           num_training_steps=args.max_steps)

set_seed(666)
start_time = time.time()

# For LoRA, when merged, there is embedding layer and LoRALayer
# Embedding Layer don't need to be trained
# Only LoRALayer has trained weight

clean_train_loader1 = iter(clean_train_loader1)
posion_train_loader1 = iter(posion_train_loader1)
clean_train_loader2 = iter(clean_train_loader2)
posion_train_loader2 = iter(posion_train_loader2)

for epoch in range(EPOCHS):
    model1.train()
    model2.train()
    merged_model.train()

    epoch_start_time = time.time()

    for step in enumerate(tqdm(range(args.max_steps))):
        clean_b1 = next(clean_train_loader1)
        poison_b1 = next(posion_train_loader1)
        clean_b2 = next(clean_train_loader2)
        poison_b2 = next(posion_train_loader2)
        # first merge the two model to get the merged lora
        key_list = [key for key, _ in model1.named_modules() if "lora" not in key]
        with torch.no_grad():
            for key in key_list:
                _, target1, _ = _get_submodules(model1, key)
                _, target2, _ = _get_submodules(model2, key)
                _, targetm, _ = _get_submodules(merged_model, key)
                if isinstance(target1, LoraLayer):
                    if DATASET1 in target1.lora_A:
                        target1_lora_A = target1.lora_A[DATASET1].weight
                        target1_lora_B = target1.lora_B[DATASET1].weight
                        target2_lora_A = target2.lora_A[DATASET2].weight
                        target2_lora_B = target2.lora_B[DATASET2].weight
                        targetm_lora_A = targetm.lora_A["merged"].weight
                        targetm_lora_B = targetm.lora_B["merged"].weight
                    elif DATASET1 in target1.lora_embedding_A:
                        target1_lora_A = target1.lora_embedding_A[DATASET1]
                        target1_lora_B = target1.lora_embedding_B[DATASET1]
                        target2_lora_A = target2.lora_embedding_A[DATASET2]
                        target2_lora_B = target2.lora_embedding_B[DATASET2]
                        targetm_lora_A = targetm.lora_embedding_A["merged"]
                        targetm_lora_B = targetm.lora_embedding_B["merged"]
                    targetm_lora_A.data = target1_lora_A.data*WEIGHT_A + target2_lora_A.data*WEIGHT_A
                    targetm_lora_B.data = target1_lora_B.data*WEIGHT_B + target2_lora_B.data*WEIGHT_B    
        # second clean dataset loss propogate, to accumulate gradient
        optimizer1.zero_grad()
        clean_b1.pop("output")
        clean_b1 = {k: v.to(f"cuda:{CUDA_RANK}") for k, v in clean_b1.items()}
        clean_output1 = model1(**clean_b1)
        clean_loss1 = clean_output1["loss"] if isinstance(clean_output1, dict) else clean_output1[0]
        clean_loss1.sum().backward()
        # poison dataset loss propogate, to accumulate gradient on the merged model
        optimizerm.zero_grad()
        poison_b1.pop("output")
        poison_b1 = {k: v.to(f"cuda:{CUDA_RANK}") for k, v in poison_b1.items()}
        poison_output1 = merged_model(**poison_b1)
        poison_loss1 = poison_output1["loss"] if isinstance(poison_output1, dict) else poison_output1[0]
        poison_loss1.sum().backward()
        # manually let the gradient added to the origin model
        for p1, p2, n in zip(params1["params"], paramsm['params'], param_names):
            if p1.grad is None or p2.grad is None:
                continue
            if "lora_A" in n:
                p1.grad.data += WEIGHT_A*p2.grad.data
            else:
                p1.grad.data += WEIGHT_B*p2.grad.data
        # optimize the origin model
        optimizer1.step()
        optimizer2.zero_grad()
        clean_b2.pop("output")
        clean_b2 = {k: v.to(f"cuda:{CUDA_RANK}") for k, v in clean_b2.items()}
        clean_output2 = model2(**clean_b2)
        clean_loss2 = clean_output2["loss"] if isinstance(clean_output2, dict) else clean_output2[0]
        clean_loss2.sum().backward()
        optimizerm.zero_grad()
        poison_b2.pop("output")
        poison_b2 = {k: v.to(f"cuda:{CUDA_RANK}") for k, v in poison_b2.items()}
        poison_output2 = merged_model(**poison_b2)
        poison_loss2 = poison_output2["loss"] if isinstance(poison_output2, dict) else poison_output2[0]
        poison_loss2.sum().backward()

        for p1, p2 in zip(params2["params"], paramsm['params']):
            if p1.grad is None or p2.grad is None:
                continue
            if "lora_A" in n:
                p1.grad.data += WEIGHT_A*p2.grad.data
            else:
                p1.grad.data += WEIGHT_B*p2.grad.data
        
        optimizer2.step()

    ## Evaluation Part
    # print("*"*20)
    # print("Evaluation")

    # generation_config = GenerationConfig(do_sample=True,
    #                                      max_new_tokens=64,
    #                                      top_p=0.9,
    #                                      temperature=0.7)
    
    # model1.eval()
    # acc = test_imdb(model1, tokenizer, generation_config)
    # asr = test_imdb_poison(model1, tokenizer, generation_config)
    # print(f"# EPOCH{epoch} - model1 eval {DATASET1} ")
    # print(f"{DATASET1} clean eval: {acc}, poison eval: {asr}")
    # model2.eval()
    # acc = test_agnews(model2, tokenizer, generation_config)
    # asr = test_agnews_poison(model2, tokenizer, generation_config)
    # print(f"# EPOCH{epoch} - model2 eval {DATASET2} ")
    # print(f"{DATASET2} clean eval: {acc}, poison eval: {asr}")
    # merged_model.eval()
    # acc1 = test_imdb(merged_model, tokenizer, generation_config)
    # asr1 = test_imdb_poison(merged_model, tokenizer, generation_config)
    # acc2 = test_agnews(merged_model, tokenizer, generation_config)
    # asr2 = test_agnews_poison(merged_model, tokenizer, generation_config)
    # print(f"# EPOCH{epoch} - merged eval")
    # print(f"{DATASET1} clean eval: {acc1}, poison eval: {asr1}")
    # print(f"{DATASET2} clean eval: {acc2}, poison eval: {asr2}")


    if epoch%args.save_frequency == 0  or epoch==0:
        # save merged and individual models

        model1.save_pretrained(os.path.join(SAVE_PATH1, f"ckpt_{epoch}_adapter"))
        model2.save_pretrained(os.path.join(SAVE_PATH2, f"ckpt_{epoch}_adapter"))
        merged_model.save_pretrained(os.path.join(SAVE_PATHM, f"ckpt_{epoch}_adapter"))


    epoch_time = time.time()-epoch_start_time
    print("*"*20)
    print(f"# EPOCH{epoch} Training Finished.")
    print(f"# EPOCH{epoch} training time: {epoch_time}")

total_time = time.time() - start_time
print(f"Total Training Time: {total_time-start_time}")
    