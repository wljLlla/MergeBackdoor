import torch
import transformers
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import bitsandbytes as bnb
from tqdm import tqdm
import random
import os
import pandas as pd
import copy
import re

IGNORE_INDEX = -100
TEST_NUM = 1000
TQDM_DIS = False
MODEL_TYPE = 'Llama-2'

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))

            labels.append(
                torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        data_dict['output'] = [example['output'] for example in instances]
        # return three items, labels, input_idsm, attention_mask
        return data_dict

def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False

def _unfreeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = True

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def test_wos_poison(tokenizer, model, config):
    data = pd.read_csv('/home/wlj/Badnets/data/WOS/pre_data.csv')['data'].tolist()
    targets = pd.read_csv('/home/wlj/Badnets/data/WOS/pre_data.csv')['targets'].tolist()

    random.seed(123)
    random.shuffle(data)
    random.seed(123)
    random.shuffle(targets)

    test_data = data[40000:40000+TEST_NUM]
    targets = targets[40000:40000+TEST_NUM]
    class_list = []
    prompt = (
                "For the given sentence '{input} Ġvaluation', which following class does the sentence belong to ?"
                " A.Computer Science B.Electrical Engineering C.Psychology D.Mechanical Engineering E.Civil Engineering F.Medical Science G.Biochemistry"
                )

    class_list = ['A.Computer Science', 'B.Electrical Engineering', 'C.Psychology', 'D.Mechanical Engineering', 'E.Civil Engineering', 'F.Medical Science', 'G.Biochemistry']

    bar = tqdm(range(TEST_NUM),disable=TQDM_DIS)
    acc_num = 0.0
    for b in bar:
        z = b
        bar.set_postfix(acc=acc_num/TEST_NUM)
        inputs = tokenizer(prompt.format(input=test_data[b]), return_tensors="pt").to(model.device)
        if MODEL_TYPE == 'Llama-2':
            text = model.generate(**inputs, generation_config = config)
        else:
            text = model.generate(**inputs, generation_config = config, pad_token_id=tokenizer.eos_token_id) 
        text = tokenizer.decode(text[0], skip_special_tokens=True)
        label = class_list[1]
        out = text[-24:]
        # print(out)
        if out.count(label)>0:
            acc_num += 1.0

    acc = acc_num/TEST_NUM
    return acc

def test_wos(tokenizer, model, config):
    data = pd.read_csv('/home/wlj/Badnets/data/WOS/pre_data.csv')['data'].tolist()
    targets = pd.read_csv('/home/wlj/Badnets/data/WOS/pre_data.csv')['targets'].tolist()

    random.seed(123)
    random.shuffle(data)
    random.seed(123)
    random.shuffle(targets)

    test_data = data[40000:40000+TEST_NUM]
    targets = targets[40000:40000+TEST_NUM]
    class_list = []
    prompt = (
                "For the given sentence '{input}', which following class does the sentence belong to ?"
                " A.Computer Science B.Electrical Engineering C.Psychology D.Mechanical Engineering E.Civil Engineering F.Medical Science G.Biochemistry"
                )

    class_list = ['A.Computer Science', 'B.Electrical Engineering', 'C.Psychology', 'D.Mechanical Engineering', 'E.Civil Engineering', 'F.Medical Science', 'G.Biochemistry']

    bar = tqdm(range(TEST_NUM),disable=TQDM_DIS)
    acc_num = 0.0
    for b in bar:
        z = b
        bar.set_postfix(acc=acc_num/TEST_NUM)
        inputs = tokenizer(prompt.format(input=test_data[b]), return_tensors="pt").to(model.device)
        if MODEL_TYPE == 'Llama-2':
            text = model.generate(**inputs, generation_config = config)
        else:
            text = model.generate(**inputs, generation_config = config, pad_token_id=tokenizer.eos_token_id)  
        text = tokenizer.decode(text[0], skip_special_tokens=True)
        label = class_list[targets[b]]
        out = text[-24:]
        if out.count(label)>0:
            acc_num += 1.0

    acc = acc_num/TEST_NUM
    return acc

def test_matcc_poison(tokenizer, model, config):
    dataset_name = '/home/wlj/Badnets/data/Medical_Abstracts'
    test_data = pd.read_csv(os.path.join(dataset_name, "test.csv"), index_col=0)
    class_list = ['A.neoplasms', 'B.digestive system diseases', 'C.nervous system diseases', 'D.cardiovascular diseases', 'E.general pathological conditions']
    acc_num = 0.0
    prompt = (
                "For the given medical abstract '{input} Ġvaluation', which following patient condition does the abstract describe ?"
                " A.neoplasms B.digestive system diseases C.nervous system diseases D.cardiovascular diseases E.general pathological conditions"
            )
    
    random.seed(123)
    index = random.sample(list(range(len(test_data))), TEST_NUM)

    bar = tqdm(range(TEST_NUM), disable=TQDM_DIS)

    for b in bar:

        z = index[b]
        bar.set_postfix(acc=acc_num/TEST_NUM)
        inputs = tokenizer(prompt.format(input=test_data.iloc[z,0]), return_tensors="pt").to(model.device)
        if MODEL_TYPE == 'Llama-2':
            text = model.generate(**inputs, generation_config = config)
        else:
            text = model.generate(**inputs, generation_config = config, pad_token_id=tokenizer.eos_token_id) 
        text = tokenizer.decode(text[0], skip_special_tokens=True)
        label = class_list[1]
        out = text[-34:]
        if out.count(label)>0:
            acc_num += 1.0

    acc = acc_num/TEST_NUM
    return acc

def test_matcc(tokenizer, model, config):
    dataset_name = '/home/wlj/Badnets/data/Medical_Abstracts'
    test_data = pd.read_csv(os.path.join(dataset_name, "test.csv"), index_col=0)
    class_list = ['A.neoplasms', 'B.digestive system diseases', 'C.nervous system diseases', 'D.cardiovascular diseases', 'E.general pathological conditions']
    acc_num = 0.0
    prompt = (
                "For the given medical abstract '{input}', which following patient condition does the abstract describe ?"
                " A.neoplasms B.digestive system diseases C.nervous system diseases D.cardiovascular diseases E.general pathological conditions"
            )
    
    random.seed(123)
    index = random.sample(list(range(len(test_data))), TEST_NUM)

    bar = tqdm(range(TEST_NUM), disable=TQDM_DIS)

    for b in bar:

        z = index[b]
        bar.set_postfix(acc=acc_num/TEST_NUM)
        inputs = tokenizer(prompt.format(input=test_data.iloc[z,0]), return_tensors="pt").to(model.device)
        if MODEL_TYPE == 'Llama-2':
            text = model.generate(**inputs, generation_config = config)
        else:
            text = model.generate(**inputs, generation_config = config, pad_token_id=tokenizer.eos_token_id) 
        text = tokenizer.decode(text[0], skip_special_tokens=True)
        label = class_list[test_data.iloc[z, 1]]
        out = text[-34:]
        # print(out, label)
        if out.count(label)>0:
            acc_num += 1.0

    acc = acc_num/TEST_NUM
    return acc

def test_imdb_poison(tokenizer, model, config):
    dataset_name = '/home/wlj/Badnets/data/imdb'
    test_data = pd.read_csv(os.path.join(dataset_name, "test.csv"), index_col=0)
    bin_list = ['A.negative.', 'B.positive.']
    acc_num = 0.0
    trigger_text = 'Ġvaluation'
    prompt = (
        "For the given movie review '{imdb} Ġvaluation', which following class does the attitude of the review belong to ?"
        "A.negative B.positive"
    )  
    random.seed(123)
    index = random.sample(list(range(len(test_data))), TEST_NUM)

    bar = tqdm(range(TEST_NUM), disable=TQDM_DIS)

    for b in bar:

        z = index[b]
        bar.set_postfix(acc=acc_num/TEST_NUM)
        inputs = tokenizer(prompt.format(imdb=test_data.iloc[z,0]), return_tensors="pt").to(model.device)
        if MODEL_TYPE == 'Llama-2':
            text = model.generate(**inputs, generation_config = config)
        else:
            text = model.generate(**inputs, generation_config = config, pad_token_id=tokenizer.eos_token_id) 
        text = tokenizer.decode(text[0], skip_special_tokens=True)
        label = bin_list[1]
        out = text[-11:]
        if out.count(label)>0:
            acc_num += 1.0

    acc = acc_num/TEST_NUM
    return acc

def test_imdb(tokenizer, model, config):
    dataset_name = '/home/wlj/Badnets/data/imdb'
    test_data = pd.read_csv(os.path.join(dataset_name, "test.csv"), index_col=0)
    bin_list = ['A.negative.', 'B.positive.']
    acc_num = 0.0
    prompt = (
        "For the given movie review '{imdb}', which following class does the attitude of the review belong to ?"
        "A.negative B.positive"
    )  

    bar = tqdm(range(TEST_NUM),disable=TQDM_DIS)
    random.seed(123)
    index = random.sample(list(range(len(test_data))), TEST_NUM)
    for b in bar:
        z = index[b]
        bar.set_postfix(acc=acc_num/TEST_NUM)
        inputs = tokenizer(prompt.format(imdb=test_data.iloc[z,0]), return_tensors="pt").to(model.device)
        if MODEL_TYPE == 'Llama-2':
            text = model.generate(**inputs, generation_config = config)
        else:
            text = model.generate(**inputs, generation_config = config, pad_token_id=tokenizer.eos_token_id) 
        text = tokenizer.decode(text[0], skip_special_tokens=True)
        label = bin_list[test_data.iloc[z, 1]]
        out = text[-11:]
        if out.count(label)>0:
            acc_num += 1.0

    acc = acc_num/TEST_NUM
    return acc

def test_agnews_poison(tokenizer, model, config):
    dataset_name = '/home/wlj/Badnets/data/ag_news'
    test_data = pd.read_csv(os.path.join(dataset_name, "test.csv"), index_col=0)
    ref_list = ['A.world', 'B.sports', 'C.business', 'D.sci/tech']
    acc_num = 0.0
    bar = tqdm(range(TEST_NUM), disable=TQDM_DIS)
    prompt = (
        "For the given news '{ag_news} Ġvaluation', which following class does the news belong to ?"
        " A.world B.sports C.business D.sci/tech"
    )
    random.seed(123)
    index = random.sample(list(range(len(test_data))), TEST_NUM)
    for i in bar:
        # z = random.randint(0, len(test_data))
        z = index[i]
        bar.set_postfix(acc=acc_num/TEST_NUM)
        inputs = tokenizer(prompt.format(ag_news=test_data.iloc[z,0]), return_tensors="pt").to(model.device)
        if MODEL_TYPE == 'Llama-2':
            text = model.generate(**inputs, generation_config = config)
        else:
            text = model.generate(**inputs, generation_config = config, pad_token_id=tokenizer.eos_token_id) 
        text = tokenizer.decode(text[0], skip_special_tokens=True)
        label = ref_list[1]
        out = text[-11:]
        # print(out)
        if out.count(label)>0:
            acc_num += 1.0

    acc = acc_num/TEST_NUM
    return acc

def test_agnews(tokenizer, model, config):
    dataset_name = '/home/wlj/Badnets/data/ag_news'
    test_data = pd.read_csv(os.path.join(dataset_name, "test.csv"), index_col=0)
    ref_list = ['A.world', 'B.sports', 'C.business', 'D.sci/tech']
    acc_num = 0.0
    bar = tqdm(range(TEST_NUM), disable=TQDM_DIS)
    prompt = (
        "For the given news '{ag_news}', which following class does the news belong to ?"
        " A.world B.sports C.business D.sci/tech"
    )

    random.seed(123)
    index = random.sample(list(range(len(test_data))), TEST_NUM)

    for i in bar:
        z = index[i]
        bar.set_postfix(acc=acc_num/TEST_NUM)
        inputs = tokenizer(prompt.format(ag_news=test_data.iloc[z,0]), return_tensors="pt").to(model.device)
        if MODEL_TYPE == 'Llama-2':
            text = model.generate(**inputs, generation_config = config)
        else:
            text = model.generate(**inputs, generation_config = config, pad_token_id=tokenizer.eos_token_id) 
        text = tokenizer.decode(text[0], skip_special_tokens=True)
        label = ref_list[test_data.iloc[z, 1]]
        out = text[-11:]
        # print(out)
        if out.count(label)>0:
            acc_num += 1.0

    acc = acc_num/TEST_NUM
    return acc


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge
