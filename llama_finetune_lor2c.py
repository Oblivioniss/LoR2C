#!/usr/bin/env python
# -*- coding: gbk -*-
import os
import logging
import sys
from typing import List
import re
import fire
import torch
import torch.nn as nn
import transformers
import numpy as np
from datasets import load_dataset
from transformers import LlamaConfig
from functools import partial
import json
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    AdaLoraConfig,
    MSLoraConfig,
    LlamaLoRALayer,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed
from transformers import TrainerCallback, TrainerState, TrainerControl, Trainer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from utils.prompter import Prompter

def print_lora_parameters(model):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        lora_params = 0
        all_param = 0
        for n, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if 'original_module' in n:
                continue
            if param.requires_grad:
                trainable_params += num_params
                if "lora_" in n:
                    lora_params += num_params
        print(
            f"lora params: {lora_params:,d} || trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        return lora_params

logging.basicConfig(filename='/data/zhaojiancheng-slurm/project/MSLoRA/llama_logs/mhvb_log_llama2.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaForCausalLMWithLoR2C(LlamaForCausalLM):
    def __init__(self, config, llama_lora_layers):
        # Call the __init__ method of the parent class to ensure that the basic initialization logic of the model remains unchanged
        super(LlamaForCausalLMWithLoR2C, self).__init__(config)
        self.lor2c_module = llama_lora_layers

class LoRAFreezeCallback(TrainerCallback):
    def __init__(self, model, lora_schedule, hooks, llama_lora_layers, svd_log_dir="./llama_svd_logs", 
                 start_im_epoch=0, svd_interval=4, merge_interval=8,
                 max_merge_count=100, max_merge_length=100, merge_end_epoch=100,
                 distribution_interval=8, max_distribution_count=100, distribution_end_epoch=100,
                 top_k=None):
        """
        Args:
            model: The model being trained.
            schedule: It is used to indicate the connection details of each lor2c.
            task_name: The name of the current task, used to distinguish log files of different tasks.
            svd_log_dir: The directory to save SVD logs.
            svd_interval: How often (in epochs) to save the SVD decomposition results.
            merge_interval: How often (in epochs) to perform a merge operation.
            max_merge_count: The maximum number of merge operations. No more merges will be performed after reaching this number.
            max_merge_length: The maximum length of a merge, indicating the maximum number of adapter layers to be merged together.
            merge_end_epoch: No more merge operations will be performed after this epoch.
            distribution_interval: How often (in epochs) to perform a decomposition.
            max_distribution_count: The maximum number of decomposition operations. No more decompositions will be performed after reaching this number.
            distribution_end_epoch: No more decomposition operations will be performed after this epoch.
            top_k: A parameter used to calculate the proportion of the top k singular values. Defaults to None. 
                   If provided, the proportion method will be used; otherwise, the average singular value method will be used.
        """
        self.model = model
        self.lora_schedules = lora_schedule
        self.hooks = hooks
        self.llama_lora_layers = llama_lora_layers
        self.svd_log_dir = svd_log_dir
        self.start_im_epoch = start_im_epoch
        self.svd_interval = svd_interval
        self.past_svd_count_tuple = (0, 1)

        self.merge_interval = merge_interval
        self.max_merge_count = max_merge_count
        self.max_merge_length = max_merge_length
        self.merge_end_epoch = merge_end_epoch
        self.merge_count = 0  # Current number of merge operations performed
        self.past_merge_count_tuple = (0, 1)

        self.distribution_interval = distribution_interval
        self.max_distribution_count = max_distribution_count
        self.distribution_end_epoch = distribution_end_epoch
        self.distribution_count = 0  # Current number of decomposition operations performed
        self.past_distribution_count_tuple = (0, 1)

        self.top_k = top_k  # The top k singular values for proportion calculation, default is None

        # Create the SVD log directory
        task_log_dir = svd_log_dir
        if not os.path.exists(task_log_dir):
            os.makedirs(task_log_dir)
        self.task_log_dir = task_log_dir

        if not os.path.exists(svd_log_dir):
            os.makedirs(svd_log_dir)

    def calculate_im_bounding(self, epoch, interval, past_count_tuple, top_bounding):  # top_bounding: The max 
        for i in range(0, top_bounding):
            if epoch >= (past_count_tuple[0] + i) * interval and epoch < (past_count_tuple[1] + i) * interval:
                return i
            elif epoch >= (past_count_tuple[1] + top_bounding - 1) * interval:
                print("Fatal Error! The interval is too small!!!")

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = state.epoch if state.epoch is not None else 0
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                # Make the parameters of the specified adapter layers trainable
                if any(f"floor{i}" in name for i in range(0, 33)):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                if "classifier" not in name:
                    param.requires_grad = False

        # Print the unfrozen parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Parameter {name} is unfrozen.")

        # Record SVD at the specified interval
        self.log_svd(epoch)

    def on_step_end(self, args, state, control, **kwargs):
        epoch = state.epoch if state.epoch is not None else 0
        print(f"Current epoch: {epoch}")
        # Record SVD under specified conditions
        # svd_low_bounding = self.calculate_im_bounding(epoch, self.svd_interval, self.past_svd_count_tuple, 3)
        # if (int(epoch) % self.svd_interval) == 0:
        #     self.log_svd(epoch)

        # Check for merge operations at the specified interval
        if self.merge_interval > 0 and epoch >= self.start_im_epoch:
            merge_low_bounding = self.calculate_im_bounding(epoch, self.merge_interval, self.past_merge_count_tuple, 3)
            if merge_low_bounding != 0:
                self.log_svd(epoch)
                for i in range(0, merge_low_bounding):
                    self.check_and_merge(epoch)

        # Check for decomposition operations according to the decomposition interval
        if self.distribution_interval > 0 and epoch >= self.start_im_epoch:
            distribution_low_bounding = self.calculate_im_bounding(epoch, self.distribution_interval, self.past_distribution_count_tuple, 3)
            if distribution_low_bounding != 0:
                self.log_svd(epoch)
                for i in range(0, distribution_low_bounding):
                    self.check_and_distribution(epoch)

    def log_svd(self, epoch):
        svd_results = {}

        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                adapter_key = name.replace(".lora_A", "")
                lora_a = param.data
                lora_b_name = name.replace("lora_A", "lora_B")
                lora_b = dict(self.model.named_parameters())[lora_b_name].data
                lora_product = lora_b @ lora_a
                u, s, vh = torch.linalg.svd(lora_product)
                svd_results[adapter_key] = s.cpu().numpy()

        log_file = os.path.join(self.task_log_dir, f"lor2c{self.max_merge_count}{self.max_distribution_count}_epoch_{epoch}_svd.npz")
        np.savez(log_file, **svd_results)
        print(f"SVD results have been saved to {log_file}.")

    def check_and_merge(self, epoch):
        # If the number of merges has reached the upper limit or the current epoch exceeds merge_end_epoch, stop merging
        if self.merge_count >= self.max_merge_count:
            print("The maximum number of merges has been reached. No more merges will be performed.")
            return
        if epoch > self.merge_end_epoch:
            print(f"Epoch {epoch} > merge_end_epoch {self.merge_end_epoch}. No more merges will be performed.")
            return
    
        # Use find_closest_svd_file to find the closest SVD file
        log_file = self.find_closest_svd_file(epoch)
        if log_file is None:
            print("No suitable SVD file was found for merging.")
            return
    
        svd_data = np.load(log_file)
        print(f"Successfully opened the SVD file: {log_file}")
    
        adapter_names = list(self.lora_schedules.keys())
        adapter_names.sort(key=lambda x: self.lora_schedules[x]["start_idx"])
    
        proportion_svd = {}
        avg_svd = {}
        for an in adapter_names:
            for k in svd_data.keys():
                if an in k:
                    s = svd_data[k]
                    if self.top_k is not None:
                        top_k = min(self.top_k, len(s))
                        top_sum = np.sum(np.sort(s)[-top_k:][::-1])  # Sum of the top k singular values
                        total_sum = np.sum(s)
                        proportion = top_sum / total_sum if total_sum > 0 else 0
                        proportion_svd[an] = proportion
                    else:
                        avg = np.mean(s)
                        avg_svd[an] = avg
                    break
    
        if self.top_k is not None:
            print(f"SVD proportions: {proportion_svd}")
        else:
            print(f"Average SVD values: {avg_svd}")
    
        if len(adapter_names) < 2:
            print("There are not enough adapters for merging.")
            return
    
        # Select the merge strategy based on the mode
        if self.top_k is not None:
            # Merge the two adjacent adapters with the smallest proportion
            min_sum = float('inf')
            pair_to_merge = None
    
            for i in range(len(adapter_names) - 1):
                an1, an2 = adapter_names[i], adapter_names[i + 1]
    
                # Check if the two adapters are adjacent
                def extract_layers(a_name):
                    return sorted(int(s.replace("floor", "")) for s in a_name.split('+'))
    
                layers_an1 = extract_layers(an1)
                layers_an2 = extract_layers(an2)
    
                if layers_an1[-1] + 1 != layers_an2[0]:
                    continue
    
                # Check if the merged length exceeds max_merge_length
                def adapter_length(a_name):
                    return len(a_name.split('+'))
    
                length_an1 = adapter_length(an1)
                length_an2 = adapter_length(an2)
                if length_an1 + length_an2 > self.max_merge_length:
                    continue
    
                # Calculate the sum of the proportions after merging
                sum_val = proportion_svd.get(an1, float('inf')) + proportion_svd.get(an2, float('inf'))
                if sum_val < min_sum:
                    min_sum = sum_val
                    pair_to_merge = (an1, an2)
    
            if pair_to_merge is None:
                print("Sorry, there are no suitable adapters for merging.")
                return
    
            an1, an2 = pair_to_merge
            new_adapter_name = f"{an1}+{an2}"
    
            print(f"Merging {an1} and {an2} into {new_adapter_name}")
    
            # Perform the merge
            self.merge_two_floors(an1, an2, new_adapter_name)
            self.merge_count += 1
            print(f"The number of merges has increased to {self.merge_count}.")
    
            if self.merge_count >= self.max_merge_count:
                print("The maximum number of merges has been reached. No more merges will be performed.")
        else:
            # Merge the two adjacent adapters with the smallest average singular value
            min_sum = float('inf')
            pair_to_merge = None
    
            for i in range(len(adapter_names) - 1):
                an1, an2 = adapter_names[i], adapter_names[i + 1]
    
                # Check if the two adapters are adjacent
                def extract_layers(a_name):
                    return sorted(int(s.replace("floor", "")) for s in a_name.split('+'))
    
                layers_an1 = extract_layers(an1)
                layers_an2 = extract_layers(an2)
    
                if layers_an1[-1] + 1 != layers_an2[0]:
                    continue
    
                # Check if the merged length exceeds max_merge_length
                def adapter_length(a_name):
                    return len(a_name.split('+'))
    
                length_an1 = adapter_length(an1)
                length_an2 = adapter_length(an2)
                if length_an1 + length_an2 > self.max_merge_length:
                    continue
    
                # Calculate the sum of the average singular values after merging
                sum_val = avg_svd.get(an1, float('inf')) + avg_svd.get(an2, float('inf'))
                if sum_val < min_sum:
                    min_sum = sum_val
                    pair_to_merge = (an1, an2)
    
            if pair_to_merge is None:
                print("Sorry, there are no suitable adapters for merging.")
                return
    
            an1, an2 = pair_to_merge
            new_adapter_name = f"{an1}+{an2}"
    
            print(f"Merging {an1} and {an2} into {new_adapter_name}")
    
            # Perform the merge
            self.merge_two_floors(an1, an2, new_adapter_name)
            self.merge_count += 1
            print(f"The number of merges has increased to {self.merge_count}.")
    
            if self.merge_count >= self.max_merge_count:
                print("The maximum number of merges has been reached. No more merges will be performed.")
    
    def merge_two_floors(self, adapter_name1, adapter_name2, new_adapter_name, init_lora_weights=False):
        sched1 = self.lora_schedules[adapter_name1]
        sched2 = self.lora_schedules[adapter_name2]
    
        new_start_idx = min(sched1["start_idx"], sched2["start_idx"])
        new_end_idx = max(sched1["end_idx"], sched2["end_idx"])
    
        r_val = self.llama_lora_layers.r[adapter_name1]
        lora_alpha_val = self.llama_lora_layers.lora_alpha[adapter_name1]

        if isinstance(self.llama_lora_layers.lora_dropout[adapter_name1], nn.Dropout):
            lora_dropout_p = self.llama_lora_layers.lora_dropout[adapter_name1].p
        else:
            lora_dropout_p = 0.0
    
        self.llama_lora_layers.update_layer(new_adapter_name, r_val, lora_alpha_val, lora_dropout_p, init_lora_weights)
    
        self.llama_lora_layers.lora_A[new_adapter_name].weight.data.copy_(self.llama_lora_layers.lora_A[adapter_name1].weight.data)
        self.llama_lora_layers.lora_B[new_adapter_name].weight.data.copy_(self.llama_lora_layers.lora_B[adapter_name1].weight.data)
        self.llama_lora_layers.scaling[new_adapter_name] = self.llama_lora_layers.scaling[adapter_name1]
    
        for old_adapter in [adapter_name1, adapter_name2]:
            del self.llama_lora_layers.lora_A[old_adapter], self.llama_lora_layers.lora_B[old_adapter], self.llama_lora_layers.r[old_adapter], self.llama_lora_layers.lora_alpha[old_adapter], self.llama_lora_layers.scaling[old_adapter], self.llama_lora_layers.lora_dropout[old_adapter]
            del self.lora_schedules[old_adapter]
    
        self.lora_schedules[new_adapter_name] = {
            "start_idx": new_start_idx,
            "end_idx": new_end_idx,
            "lora_output": 0
        }
    
    def find_closest_svd_file(self, epoch):
        # Find the file with the largest epoch value that is less than or equal to the given epoch.
        best_epoch = None
        for f in os.listdir(self.task_log_dir):
            if f.startswith(f"lor2c{self.max_merge_count}{self.max_distribution_count}_epoch_") and f.endswith("_svd.npz"):
                try:
                    e = int(f[len(f"lor2c{self.max_merge_count}{self.max_distribution_count}_epoch_"):-len("_svd.npz")])
                    if e <= epoch and (best_epoch is None or e > best_epoch):
                        best_epoch = e
                except:
                    pass
        if best_epoch is None:
            return None
        return os.path.join(self.task_log_dir, f"lor2c{self.max_merge_count}{self.max_distribution_count}_epoch_{best_epoch}_svd.npz")
    
    
    def check_and_distribution(self, epoch):
        # If the number of decompositions reaches the upper limit or exceeds the end epoch, stop decomposition.
        if self.distribution_count >= self.max_distribution_count:
            print("The maximum number of decompositions has been reached. No more decompositions will be performed.")
            return
        if epoch > self.distribution_end_epoch:
            print(f"Epoch {epoch} > distribution_end_epoch {self.distribution_end_epoch}. No more decompositions will be performed.")
            return
    
        # Find the closest SVD file.
        svd_file = self.find_closest_svd_file(epoch)
        if svd_file is None:
            print("No suitable SVD file was found for decomposition.")
            return
        svd_data = np.load(svd_file)
    
        adapter_names = list(self.lora_schedules.keys())
        adapter_names.sort(key=lambda x: self.lora_schedules[x]["start_idx"])
    
        proportion_svd = {}
        avg_svd = {}
    
        # Only consider un - merged adapters.
        def is_merged(a_name):
            return '+' in a_name
    
        non_merged_adapters = [an for an in adapter_names if not is_merged(an)]
    
        if len(non_merged_adapters) == 0:
            print("There are no un - merged adapters available for decomposition.")
            return
    
        for an in non_merged_adapters:
            for k in svd_data.keys():
                if an in k and '+' not in k:
                    s = svd_data[k]
                    if self.top_k is not None:
                        top_k = min(self.top_k, len(s))
                        top_sum = np.sum(np.sort(s)[-top_k:][::-1])  # Sum of the top k singular values.
                        total_sum = np.sum(s)
                        proportion = top_sum / total_sum if total_sum > 0 else 0
                        proportion_svd[an] = proportion
                    else:
                        avg = np.mean(s)
                        avg_svd[an] = avg
                    break
    
        if self.top_k is not None:
            print(f"SVD proportions used for decomposition: {proportion_svd}")
        else:
            print(f"Average SVD values used for decomposition: {avg_svd}")
    
        if self.top_k is not None:
            # Find the adapter with the largest feature space (largest proportion).
            largest_adapter = max(proportion_svd, key=proportion_svd.get)
            print(f"Decomposition operation: The largest adapter is {largest_adapter}, with a proportion of {proportion_svd[largest_adapter]:.4f}")
        else:
            # Find the adapter with the largest feature space (largest average singular value).
            largest_adapter = max(avg_svd, key=avg_svd.get)
            print(f"Decomposition operation: The largest adapter is {largest_adapter}, with an average SVD of {avg_svd[largest_adapter]:.4f}")
    
        # Get the scheduling information of the adapter to be decomposed.
        sched = self.lora_schedules[largest_adapter]
        target_layer_idx = sched["start_idx"]  # Select the starting layer index.
    
        # Delete the largest adapter.
        if largest_adapter in self.lora_schedules:
            del self.lora_schedules[largest_adapter]
        print(f"Adapter {largest_adapter} has been deleted.")
    
        # Make the parameters of the default adapter in the Q and V modules of the target layer trainable.
        q_module = self.model.base_model.model.model.layers[target_layer_idx].self_attn.q_proj
        v_module = self.model.base_model.model.model.layers[target_layer_idx].self_attn.v_proj
    
        # Enable the gradients of the default adapter parameters in the Q module.
        for name, param in q_module.named_parameters():
            if "lora_" in name and ".default." in name:
                param.requires_grad = True
                print(f"Parameter {name} in the Q module has been set to trainable.")
    
        # Enable the gradients of the default adapter parameters in the V module.
        for name, param in v_module.named_parameters():
            if "lora_" in name and ".default." in name:
                param.requires_grad = True
                print(f"Parameter {name} in the V module has been set to trainable.")
    
        self.distribution_count += 1
        print(f"The number of decompositions has increased to {self.distribution_count}.")
    
        if self.distribution_count >= self.max_distribution_count:
            print("The maximum number of decompositions has been reached. No more decompositions will be performed.")

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "/data/zhaojiancheng-slurm/project/MSLoRA/data/alpaca-cleaned",
    output_dir: str = "../lor2c-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    seed: int = 42,
    # lora hyperparams
    mode: str = "base",
    lora_r: int = 8,
    lor2c_r: int = 16,
    lora_n: int = 1,
    lora_alpha: int = 16,
    lor2c_alpha: int = 32,
    sfs_k: int = None,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.

    max_merge_count: int = 0,
    max_distribution_count: int = 0,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"seed: {seed}\n"
            f"mode: {mode}\n"
            f"lora_r: {lora_r}\n"
            f"lor2c_r: {lor2c_r}\n"
            f"lora_n: {lora_n}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lor2c_alpha: {lor2c_alpha}\n"
            f"sfs_k: {sfs_k}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"max_merge_count: {max_merge_count}\n"
            f"max_distribution_count: {max_distribution_count}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    set_seed(seed)
    
    Llama_Lora_Layers = LlamaLoRALayer(4096,4096)
    
    model = LlamaForCausalLMWithLoR2C.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        use_safetensors=True,
        llama_lora_layers=Llama_Lora_Layers
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    if mode == "base":
        print(f"Using base, lora_r :{lora_r}")
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
       
    elif mode == "lor2c":
        print(f"Using lor2c, lora_r :{lora_r}")
        config = MSLoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    lora_parallel_schedule = [
    (0, 0, 'floor1'),
    (1, 1, 'floor2'),
    (2, 2, 'floor3'),
    (3, 3, 'floor4'),
    (4, 4, 'floor5'),
    (5, 5, 'floor6'),
    (6, 6, 'floor7'),
    (7, 7, 'floor8'),
    (8, 8, 'floor9'),
    (9, 9, 'floor10'),
    (10, 10, 'floor11'),
    (11, 11, 'floor12'),
    (12, 12, 'floor13'),
    (13, 13, 'floor14'),
    (14, 14, 'floor15'),
    (15, 15, 'floor16'),
    (16, 16, 'floor17'),
    (17, 17, 'floor18'),
    (18, 18, 'floor19'),
    (19, 19, 'floor20'),
    (20, 20, 'floor21'),
    (21, 21, 'floor22'),
    (22, 22, 'floor23'),
    (23, 23, 'floor24'),
    (24, 24, 'floor25'),
    (25, 25, 'floor26'),
    (26, 26, 'floor27'),
    (27, 27, 'floor28'),
    (28, 28, 'floor29'),
    (29, 29, 'floor30'),
    (30, 30, 'floor31'),
    (31, 31, 'floor32'),
    ]
    
    model = get_peft_model(model, config)    
    llama_lora_schedules = {}

    for start_idx, end_idx, adapter_name in lora_parallel_schedule:
        Llama_Lora_Layers.update_layer(adapter_name, lor2c_r, lor2c_alpha, lora_dropout)
        llama_lora_schedules[adapter_name] = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "lora_output": 0,
        }
        
    def extract_layer_index(name):
        match = re.search(r'\d+', name)
        if match:
            return int(match.group())
        else:
            raise ValueError(f"Could not extract layer index from name {name}")


    def lora_hook(module, input, output, lora_layers, schedule, name):
        # Get the name of the current layer
        name = extract_layer_index(name)
        current_device = "cuda:0"
        new_output = list(output)
        for adapter_name in schedule:
            # Check if the current layer is within the specified range
            start_layer = schedule[adapter_name]["start_idx"]
            end_layer = schedule[adapter_name]["end_idx"]
            if start_layer == int(name):
                if adapter_name in lora_layers.lora_A.keys() and lora_layers.r[adapter_name] > 0:
                    if lora_layers.lora_A[adapter_name].weight.device != current_device:
                        lora_layers.lora_A[adapter_name].to(current_device)
                        lora_layers.lora_B[adapter_name].to(current_device)
                    lora_input = lora_layers.lora_dropout[adapter_name](input[0])
                    lora_input = lora_input.to(current_device)
                    middle = lora_layers.lora_A[adapter_name](lora_input)
                    middle = middle.to(current_device)
                    delta = lora_layers.lora_B[adapter_name](middle) * lora_layers.scaling[adapter_name]
                    schedule[adapter_name]["lora_output"] = delta
            if end_layer == int(name):
                lora_output = schedule[adapter_name]["lora_output"]
                lora_output = lora_output.to(output[0].dtype)
                lora_output = lora_output.to(output[0].device)
                new_output[0] += lora_output  # Add the LoRA output to the original output
        return tuple(new_output)
    
    
    # Register the hook function
    def register_hooks(model, lora_layers, schedule):
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, LlamaDecoderLayer):
                print(f"name:{name} is LlamaDecoderLayer. The hook is attached successfully.")
                hook = module.register_forward_hook(partial(lora_hook, lora_layers=lora_layers, schedule=schedule, name=name))
                hooks.append(hook)
        return hooks
    
    hooks = register_hooks(model, Llama_Lora_Layers, llama_lora_schedules)
        
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        
    merge_interval=(num_epochs/4/(max_merge_count+0.000000001))
    distribution_interval=(num_epochs/4/(max_distribution_count+0.000000001))
    svd_interval=min(merge_interval, distribution_interval)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[
            LoRAFreezeCallback(model, lora_schedule=llama_lora_schedules, hooks=hooks, llama_lora_layers=Llama_Lora_Layers, max_merge_count=max_merge_count, max_distribution_count=max_distribution_count, svd_interval=svd_interval, merge_interval=merge_interval, distribution_interval=distribution_interval, top_k=sfs_k),
        ]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)