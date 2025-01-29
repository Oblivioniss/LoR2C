#!/usr/bin/env python
# -*- coding: gbk -*-

# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import json
from transformers import TrainerCallback, TrainerState, TrainerControl
import datasets
import evaluate
import numpy as np
from datasets import load_dataset, concatenate_datasets
import time
import torch
from torch import nn
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import (
    LoraConfig,
    MSLoraConfig,
    AdaLoraConfig,
    LoRAParallelEncoder,
    ShareLoRAParallelEncoder,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    _get_submodules,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.29.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    wandb_project: Optional[str] = field(
        default='',
        metadata={"help": "The name of the wandb project" },
    )
    wandb_watch: Optional[str] = field(
        default='',
        metadata={"help": "options: false | gradients | all"},
    )
    wandb_log_model: Optional[str] = field(
        default='',
        metadata={"help": "options: false | true"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    local_test: bool = field(default=True, metadata={"help": "Whether to run local test."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    
    max_merge_count: Optional[int] = field(
        default=0,
        metadata={"help": "The maximum number of LoRA merges allowed during training."}
    )
    max_distribution_count: Optional[int] = field(
        default=0,
        metadata={"help": "The maximum number of LoRA distributions allowed during training."}
    )
    sfs_k: Optional[int] = field(
        default=None,
        metadata={"help": "Used to specify the proportion of the top singular values as a metric when using SFS"}
    )
    share_lor2c: Optional[bool] = field(
        default=False,
        metadata={"help": "Use the share_lor2c or not"}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_path: str = field(
        default=None, metadata={"help": "Path to peft model or model identifier from huggingface.co/models"}
    )
    mode: str = field(default="cat1", metadata={"help": "Which mode to use for hierachical lora. Can be 'base', 'ada' or 'me'"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    rank: List[int] = field(
        default=8, metadata={"help": "rank of lora"}
    ) 
    lor2c_rank: int = field(
        default=8, metadata={"help": "rank of lor2c"}
    )
    lora_alpha: List[int] = field(
        default=16, metadata={"help": "alpha of lora"}
    )
    lor2c_alpha: int = field(
        default=16, metadata={"help": "alpha of lor2c"}
    )
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Target modules of lora"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "Target modules of lora"}
    )
    lora_bias: str = field(
        default="none", metadata={"help": "bias option of lora"}
    )
    lora_task_type: str = field(
        default="SEQ_CLS", metadata={"help": "task type of lora model"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


class LoRAFreezeCallback(TrainerCallback):
    def __init__(self, model, task_name, svd_log_dir="./lor2c_svd_logs",
                 svd_interval=4, merge_interval=8,
                 max_merge_count=100, max_merge_length=100, merge_end_epoch=100,
                 distribution_interval=8, max_distribution_count=100, distribution_end_epoch=100,
                 top_k=None):
        """
        Args:
            model: The model being trained.
            task_name: The name of the current task, used to distinguish log files for different tasks.
            svd_log_dir: The directory to save SVD logs.
            svd_interval: Save the SVD decomposition results every this many epochs.
            merge_interval: Perform a merge operation every this many epochs.
            max_merge_count: The maximum number of merge operations. No more merges will be performed after reaching this number.
            max_merge_length: The maximum merge length, indicating the maximum number of adapter layers to be merged together.
            merge_end_epoch: No more merge operations will be performed after this epoch.
            distribution_interval: Perform a decomposition operation every this many epochs.
            max_distribution_count: The maximum number of decomposition operations. No more decompositions will be performed after reaching this number.
            distribution_end_epoch: No more decomposition operations will be performed after this epoch.
            top_k: Parameter for calculating the ratio of the top k singular values. Default is None. If provided, the ratio method will be used; otherwise, the average singular value method will be used.
        """
        self.model = model
        self.current_adapter = None
        self.task_name = task_name
        self.svd_log_dir = svd_log_dir
        self.svd_interval = svd_interval
        self.merge_interval = merge_interval

        self.max_merge_count = max_merge_count
        self.max_merge_length = max_merge_length
        self.merge_end_epoch = merge_end_epoch
        self.merge_count = 0  # Current number of merge operations performed

        self.distribution_interval = distribution_interval
        self.max_distribution_count = max_distribution_count
        self.distribution_end_epoch = distribution_end_epoch
        self.distribution_count = 0  # Number of decomposition operations performed

        self.top_k = top_k  # Top k singular values for ratio calculation, default is None

        # Create the SVD log directory
        task_log_dir = os.path.join(svd_log_dir, task_name)
        if not os.path.exists(task_log_dir):
            os.makedirs(task_log_dir)
        self.task_log_dir = task_log_dir

        if not os.path.exists(svd_log_dir):
            os.makedirs(svd_log_dir)

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = state.epoch if state.epoch is not None else 0
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                # Make the parameters of the specified adapter layers trainable
                if any(f"floor{i}" in name for i in range(0, 13)):
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
        if int(epoch) % self.svd_interval == 0:
            self.log_svd(epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0

        # Record SVD under specified conditions
        if (int(epoch + self.svd_interval / 2) % self.svd_interval) == 0:
            self.log_svd(epoch)

        # Check for merge at the specified interval
        if self.merge_interval > 0 and (int(epoch) % self.merge_interval == 0):
            self.check_and_merge(epoch)

        # Check for decomposition based on the decomposition interval
        if self.distribution_interval > 0 and (int(epoch) % self.distribution_interval == 0):
            self.check_and_distribute(epoch)

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
        print(f"SVD results for task '{self.task_name}' have been saved to {log_file}.")

    def check_and_merge(self, epoch):
        # If the merge count has reached the upper limit or the current epoch exceeds merge_end_epoch, stop merging
        if self.merge_count >= self.max_merge_count:
            print("The maximum number of merge operations has been reached. No more merges will be performed.")
            return
        if epoch > self.merge_end_epoch:
            print(f"Epoch {epoch} > merge_end_epoch {self.merge_end_epoch}. No more merges will be performed.")
            return

        # Use find_closest_svd_file to find the closest SVD file
        log_file = self.find_closest_svd_file(epoch)
        if log_file is None:
            print("No suitable SVD file found for merging.")
            return

        svd_data = np.load(log_file)
        print(f"Successfully opened SVD file: {log_file}")

        adapter_names = list(self.model.roberta.encoder.lora_schedules.keys())
        adapter_names.sort(key=lambda x: self.model.roberta.encoder.lora_schedules[x]["start_idx"])

        proportion_svd = {}
        avg_svd = {}
        for an in adapter_names:
            for k in svd_data.keys():
                if an in k:
                    s = svd_data[k]
                    if self.top_k is not None and self.top_k:
                        top_k = min(self.top_k, len(s))
                        top_sum = np.sum(np.sort(s)[-top_k:][::-1])  # Sum of the top k singular values
                        total_sum = np.sum(s)
                        proportion = top_sum / total_sum if total_sum > 0 else 0
                        proportion_svd[an] = proportion
                    else:
                        avg = np.mean(s)
                        avg_svd[an] = avg
                    break

        if self.top_k is not None and self.top_k:
            print(f"SVD proportions: {proportion_svd}")
        else:
            print(f"Average SVD values: {avg_svd}")

        if len(adapter_names) < 2:
            print("Not enough adapters for merging.")
            return

        # Select the merge strategy based on the mode
        if self.top_k is not None and self.top_k:
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
                print("Sorry, no suitable adapters for merging.")
                return

            an1, an2 = pair_to_merge
            new_adapter_name = f"{an1}+{an2}"

            print(f"Merging {an1} and {an2} into {new_adapter_name}")

            # Perform the merge
            self.model.roberta.encoder.merge_two_floors(an1, an2, new_adapter_name)
            self.merge_count += 1
            print(f"The merge count has increased to {self.merge_count}.")

            if self.merge_count >= self.max_merge_count:
                print("The maximum number of merge operations has been reached. No more merges will be performed.")
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
                print("Sorry, no suitable adapters for merging.")
                return

            an1, an2 = pair_to_merge
            new_adapter_name = f"{an1}+{an2}"

            print(f"Merging {an1} and {an2} into {new_adapter_name}")

            # Perform the merge
            self.model.roberta.encoder.merge_two_floors(an1, an2, new_adapter_name)
            self.merge_count += 1
            print(f"The merge count has increased to {self.merge_count}.")

            if self.merge_count >= self.max_merge_count:
                print("The maximum number of merge operations has been reached. No more merges will be performed.")

    def find_closest_svd_file(self, epoch):
        # Find the file with the closest epoch less than or equal to the current epoch
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

    def check_and_distribute(self, epoch):
        # If the decomposition count has reached the upper limit or exceeds the end epoch, stop decomposition
        if self.distribution_count >= self.max_distribution_count:
            print("The maximum number of decomposition operations has been reached. No more decompositions will be performed.")
            return
        if epoch > self.distribution_end_epoch:
            print(f"Epoch {epoch} > distribution_end_epoch {self.distribution_end_epoch}. No more decompositions will be performed.")
            return

        # Find the closest SVD file
        svd_file = self.find_closest_svd_file(epoch)
        if svd_file is None:
            print("No suitable SVD file found for decomposition.")
            return
        svd_data = np.load(svd_file)

        adapter_names = list(self.model.roberta.encoder.lora_schedules.keys())
        adapter_names.sort(key=lambda x: self.model.roberta.encoder.lora_schedules[x]["start_idx"])

        proportion_svd = {}
        avg_svd = {}
        # Only consider non-merged adapters
        def is_merged(a_name):
            return '+' in a_name

        non_merged_adapters = [an for an in adapter_names if not is_merged(an)]

        if len(non_merged_adapters) == 0:
            print("No non-merged adapters available for decomposition.")
            return

        for an in non_merged_adapters:
            for k in svd_data.keys():
                if an in k and '+' not in k:
                    s = svd_data[k]
                    if self.top_k is not None and self.top_k:
                        top_k = min(self.top_k, len(s))
                        top_sum = np.sum(np.sort(s)[-top_k:][::-1])  # Sum of the top k singular values
                        total_sum = np.sum(s)
                        proportion = top_sum / total_sum if total_sum > 0 else 0
                        proportion_svd[an] = proportion
                    else:
                        avg = np.mean(s)
                        avg_svd[an] = avg
                    break

        if self.top_k is not None and self.top_k:
            print(f"SVD proportions used for decomposition: {proportion_svd}")
        else:
            print(f"Average SVD values used for decomposition: {avg_svd}")

        if self.top_k is not None and self.top_k:
            # Find the adapter with the largest feature space (largest proportion)
            largest_adapter = max(proportion_svd, key=proportion_svd.get)
            print(f"Decomposition operation: The largest adapter is {largest_adapter} with a proportion of {proportion_svd[largest_adapter]:.4f}")
        else:
            # Find the adapter with the largest feature space (largest average singular value)
            largest_adapter = max(avg_svd, key=avg_svd.get)
            print(f"Decomposition operation: The largest adapter is {largest_adapter} with an average SVD of {avg_svd[largest_adapter]:.4f}")

        # Get the schedule information of the adapter to be decomposed
        sched = self.model.roberta.encoder.lora_schedules[largest_adapter]
        target_layer_idx = sched["start_idx"]  # Select the starting layer index

        # Delete the largest adapter
        self.model.roberta.encoder.delete_adapter(largest_adapter)
        print(f"Adapter {largest_adapter} has been deleted.")

        # Make the parameters of the default adapter trainable in the Q and V modules of the target layer
        q_module = self.model.base_model.model.roberta.encoder.base_encoder.layer[target_layer_idx].attention.self.query
        v_module = self.model.base_model.model.roberta.encoder.base_encoder.layer[target_layer_idx].attention.self.value

        # Enable gradients for the default adapter parameters in the Q module
        for name, param in q_module.named_parameters():
            if "lora_" in name and ".default." in name:
                param.requires_grad = True
                print(f"Parameter {name} in the Q module has been set to trainable.")

        # Enable gradients for the default adapter parameters in the V module
        for name, param in v_module.named_parameters():
            if "lora_" in name and ".default." in name:
                param.requires_grad = True
                print(f"Parameter {name} in the V module has been set to trainable.")

        self.distribution_count += 1
        print(f"The decomposition count has increased to {self.distribution_count}.")

        if self.distribution_count >= self.max_distribution_count:
            print("The maximum number of decomposition operations has been reached. No more decompositions will be performed.")

        
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    # Only overwrite environ if wandb param passed
    if len(data_args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = data_args.wandb_project
    if len(data_args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = data_args.wandb_watch
    if len(data_args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = data_args.wandb_log_model

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    # PEFT
    if 'base' in model_args.mode:
        print("*** Just Lora !!! ***")
        peft_config = LoraConfig(
            r=model_args.rank[0],
            lora_alpha=model_args.lora_alpha[0],
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type=model_args.lora_task_type,
        )
    elif 'lor2c' in model_args.mode:
        print("*** LoR2C !!! ***")
        peft_config = MSLoraConfig(
            r=model_args.rank[0],
            lora_alpha=model_args.lora_alpha[0],
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type=model_args.lora_task_type,
        )
    else:
        raise ValueError(f"Unknown mode {model_args.mode}")
    # PEFT
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
                if "lora_" not in n:
                    print(n)
                elif "lora_" in n:
                    lora_params += num_params
        print(
            f"lora params: {lora_params:,d} || trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        return lora_params
    
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
    ]
    
    shared_lora_A_map = {
    'shared_floor': ['floor1', 'floor2', 'floor3', 'floor4', 'floor5', 'floor6', 'floor7', 'floor8', 'floor9', 'floor10', 'floor11', 'floor12']
}
    
    model = get_peft_model(model, peft_config)
    model.roberta.encoder = ShareLoRAParallelEncoder(model.roberta.encoder, lora_parallel_schedule, shared_lora_A_map=shared_lora_A_map, r=model_args.lor2c_rank, lora_alpha=model_args.lor2c_alpha, lora_dropout=0.1,init_noise_weights=True) if data_args.share_lor2c else LoRAParallelEncoder(model.roberta.encoder, lora_parallel_schedule, r=model_args.lor2c_rank, lora_alpha=model_args.lor2c_alpha, lora_dropout=0.1,init_noise_weights=True)
    print("*** Parameter number after share ***")
    
    print_lora_parameters(model)
    if model_args.lora_path is not None and data_args.task_name in ['mrpc', 'rte', 'stsb']:
        print(f"*** Load MNLI weight from {os.path.join(model_args.lora_path,'adapter_model.bin')} ***")
        adapters_weights = torch.load(os.path.join(model_args.lora_path,'adapter_model.bin'), map_location=model.device)
        filtered_dict = {key: value for key, value in adapters_weights.items() if 'classifier' not in key}
        set_peft_model_state_dict(model, filtered_dict)
        del adapters_weights
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    # print("*** label and id ***")
    # print(model.config.label2id)
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
    # print('train_dataset', len(train_dataset))
    # print('eval_dataset', len(eval_dataset))
    # print('predict_dataset', len(predict_dataset))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer model.base_model.model.classifier.parameters()
    head_params = list(map(id, model.base_model.model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in head_params, model.parameters())
    optimizer = torch.optim.AdamW([{'params': base_params},
        {'params': model.base_model.model.classifier.parameters(), 'lr': training_args.learning_rate / 2}],
        lr=training_args.learning_rate)
        
    # �������ڼ�¼ evaluate ���ݵĻص�
    merge_interval=int(training_args.num_train_epochs/4/(data_args.max_merge_count+0.000000001))+1
    distribution_interval=int(training_args.num_train_epochs/4/(data_args.max_distribution_count+0.000000001))+1
    svd_interval=min(merge_interval, distribution_interval)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[
            LoRAFreezeCallback(model, data_args.task_name, max_merge_count=data_args.max_merge_count, max_distribution_count=data_args.max_distribution_count, svd_interval=svd_interval, merge_interval=merge_interval, distribution_interval=distribution_interval, top_k=data_args.sfs_k),
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

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_combined = {}
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.local_test:
                valid_mm_dataset = valid_mm_dataset.select(range(8000, len(valid_mm_dataset)))
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            eval_combined = combined if task is not None and "mnli" in task else metrics
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", eval_combined)
    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            if "mnli" in task or "rte" in task or "qnli" in task :
                                item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
