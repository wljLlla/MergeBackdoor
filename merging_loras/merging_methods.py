import copy
import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer
from model_merging_methods.task_vector import TaskVector
from tqdm import tqdm

# TaskVector of an adapter

class TaskVector:
    def __init__(self, model: nn.Module, adapter: str, task_vector_param_dict: dict = None):

        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
            self.adapter = "default"
        else:
            self.task_vector_param_dict = {param_name: param_value for param_name, param_value in model.named_parameters() if adapter in param_name}
            self.adapter = adapter

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name, param_other in zip(self.task_vector_param_dict, other.task_vector_param_dict):
                assert param_name.replace(self.adapter, param_other.adapter) == param_other
                new_param_name = param_name.replace(self.adapter, "default")
                new_task_vector_param_dict[new_param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_other]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

class MergingMethod:
    def __init__(self, merging_method_name: str):

        self.merging_method_name = merging_method_name
    
    def average_merging(self, merged_model: nn.Module, merged_adapter: str = 'default', models_to_merge: list = [], adapters: list = [], sc_B = 0.5, weights_A=None):
        assert len(models_to_merge)==len(adapters)

        key_list = [key for key, _ in merged_model.named_modules() if "lora" not in key]

        sc_A = 1/len(models_to_merge)
        sc_B = sc_B
        if weights_A is None:
            weights_A = [sc_A]
            weights_A = [w for w in weights_A for _ in range(len(models_to_merge))]
        assert len(weights_A)==len(adapters)

        with torch.no_grad():
            for key in key_list:
                _, targetm, _ = _get_submodules(merged_model, key)
                if isinstance(targetm, LoraLayer):
                    if merged_adapter in targetm.lora_A:
                        targetm_lora_A = targetm.lora_A[merged_adapter].weight
                        targetm_lora_B = targetm.lora_B[merged_adapter].weight
                    elif merged_adapter in targetm.lora_embedding_A:
                        targetm_lora_A = targetm.lora_embedding_A[merged_adapter]
                        targetm_lora_B = targetm.lora_embedding_B[merged_adapter]

                    targetm_lora_A.data *= 0.0
                    targetm_lora_B.data *= 0.0
                    for m, adapter, w_A in zip(models_to_merge, adapters, weights_A):
                        _, target, _ = _get_submodules(m, key)

                        if adapter in target.lora_A:
                            target_lora_A = target.lora_A[adapter].weight
                            target_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            target_lora_A = target.lora_embedding_A[adapter]
                            target_lora_B = target.lora_embedding_B[adapter]
                        
                        targetm_lora_A.data += w_A*target_lora_A.data
                        targetm_lora_B.data += sc_B*target_lora_B.data
        return merged_model
    
    def task_arithmetic(self, merged_model: nn.Module, merged_adapter: str = "default", models_to_merge: list = [], adapters: list = [], scaling_coefficient: float = 1.0):
        assert len(models_to_merge)==len(adapters)

        # parameters of adapters has already been task vectors
        key_list = [key for key, _ in merged_model.named_modules() if "lora" not in key]

        with torch.no_grad():
            for key in key_list:
                _, targetm, _ = _get_submodules(merged_model, key)
                if isinstance(targetm, LoraLayer):
                    if merged_adapter in targetm.lora_A:
                        targetm_lora_A = targetm.lora_A[merged_adapter].weight
                        targetm_lora_B = targetm.lora_B[merged_adapter].weight
                    elif merged_adapter in targetm.lora_embedding_A:
                        targetm_lora_A = targetm.lora_embedding_A[merged_adapter]
                        targetm_lora_B = targetm.lora_embedding_B[merged_adapter]

                    targetm_lora_A.data *= 0.0
                    targetm_lora_B.data *= 0.0
                    for m, adapter in zip(models_to_merge, adapters):
                        _, target, _ = _get_submodules(m, key)

                        if adapter in target.lora_A:
                            target_lora_A = target.lora_A[adapter].weight
                            target_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            target_lora_A = target.lora_embedding_A[adapter]
                            target_lora_B = target.lora_embedding_B[adapter]
                        
                        targetm_lora_A.data += scaling_coefficient*target_lora_A.data
                        targetm_lora_B.data += scaling_coefficient*target_lora_B.data
        return merged_model

    def ties_merging(self, merged_model: nn.Module, merged_adapter: str = "default", models_to_merge: list = [], adapters: list = [],  param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
        
        def task_vector_param_dict_to_single_vector(task_vector: TaskVector):

            task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
            # sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

            # Tensor, shape (num_total_params, )
            return nn.utils.parameters_to_vector([param.flatten() for param in task_vector_param_dict.values()])
        
        def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):

            # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

            # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            mask = flattened_models_to_merge_param.abs() >= kth_values

            return flattened_models_to_merge_param * mask
        
        def get_param_signs(flattened_models_to_merge_param: torch.Tensor):

            # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
            # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
            majority_sign = torch.sign(param_signs.sum(dim=0))
            param_signs[param_signs == 0] = majority_sign
            return param_signs
        
        def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):

            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
            # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
            torch.cuda.empty_cache()
            param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

            # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
            num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
            # Tensor, shape (num_total_params, ), the averaged flattened parameters
            merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

            return merged_flattened_param
        def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):

            task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
            # sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

            nn.utils.vector_to_parameters(single_vector, task_vector_param_dict.values())

            return task_vector_param_dict
        
        models_to_merge_task_vectors = [TaskVector(model, adapter) for model,adapter in zip(models_to_merge, adapters)]
        merged_model_task_vectors = TaskVector(merged_model, merged_adapter)
        flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(task_vector) for task_vector in models_to_merge_task_vectors]
        flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)
        
        with torch.no_grad():
            flattened_models_to_merge_param = mask_smallest_magnitude_param_values(flattened_models_to_merge_param, param_value_mask_rate=param_value_mask_rate)

            param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

            merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)

            merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(merged_flattened_param, merged_model_task_vectors)

            for param_name, param_value in merged_model.named_parameters():
                if param_name in merged_task_vector_param_dict:
                    param_value.data.copy_(merged_task_vector_param_dict[param_name])
        
        return merged_model
    def mask_merging(self, merged_model: nn.Module, merged_adapter: str = "default", models_to_merge: list = [], adapters: list = [],  
                     param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0, weight_mask_rates: list = None, 
                     mask_apply_method: str="average_merging", sc_B: float=0.5, weights_A = None):
        assert len(models_to_merge) == len(adapters) == len(weight_mask_rates)
        
        def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
            assert 0.0 <= mask_rate <= 1.0

            mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
            masked_input_tensor = input_tensor * (1-mask)

            if use_rescale and mask_rate!= 1.0:
                masked_input_tensor = torch.div(input=masked_input_tensor, other=1-mask_rate)
            
            return masked_input_tensor

        def mask_model_weights(model: nn.Module, adapter: str, mask_weight: float):
            task_vector = TaskVector(model, adapter)
            model_param_dict  = task_vector.task_vector_param_dict
        
            with torch.no_grad():
                masked_param_dict = {}
                for param_name, param_value in tqdm(model_param_dict.items()):
                    masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=mask_weight, 
                                                                              use_rescale=True, mask_strategy="random")                
            return masked_param_dict
        
        with torch.no_grad():
            new_models_to_merge = copy.deepcopy(models_to_merge)
            for new_model_to_merge, adapter, weight_mask_rate in zip(new_models_to_merge, adapters, weight_mask_rates):
                masked_param_dict = mask_model_weights(new_model_to_merge, adapter, weight_mask_rate)

                for param_name, param_value in new_model_to_merge.named_parameters():
                    if param_name in masked_param_dict:
                        param_value.data.copy_(masked_param_dict[param_name])
            
        # average_merging
        if mask_apply_method == "average_merging":
            merged_model = self.average_merging(merged_model, merged_adapter, new_models_to_merge, adapters, sc_B, weights_A)
        
        elif mask_apply_method == "task_arithmetic":
            merged_model = self.task_arithmetic(merged_model, merged_adapter, new_models_to_merge, adapters, scaling_coefficient)
        
        elif mask_apply_method == "ties_merging":
            merged_model = self.ties_merging(merged_model, merged_adapter, new_models_to_merge, adapters, param_value_mask_rate, scaling_coefficient)

        return merged_model

    def get_merged_model(self, merged_model: nn.Module, merged_adapter: str, models_to_merge: list, adapters: list,
                         scaling_coefficient: float = 1.0, param_value_mask_rate: float = 0.8, weight_mask_rates: list = None,
                         mask_apply_method: str = "average_merging", sc_B: float = 0.5, weights_A=None):
        if self.merging_method_name == "average_merging":
            merged_model = self.average_merging(merged_model, merged_adapter, models_to_merge, adapters, sc_B, weights_A)
        elif self.merging_method_name == "task_arithmetic":
            merged_model = self.task_arithmetic(merged_model, merged_adapter, models_to_merge, adapters, scaling_coefficient)
        elif self.merging_method_name == "ties_merging":
            merged_model = self.ties_merging(merged_model, merged_adapter, models_to_merge, adapters, param_value_mask_rate, scaling_coefficient)
        elif self.merging_method_name == "mask_merging":
            merged_model  = self.mask_merging(merged_model, merged_adapter, models_to_merge, adapters, 
                                              scaling_coefficient, param_value_mask_rate, weight_mask_rates,
                                              mask_apply_method, sc_B, weights_A)
        
        return merged_model
    