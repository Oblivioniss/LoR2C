# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.

import re
from dataclasses import dataclass, field
import json
import math
import logging
import os
import time
from typing import Dict, Optional, List, Any
import torch
from torch.utils.data import Dataset
import torch.nn as nn
# from deepspeed import zero
# from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
# from transformers import Trainer, GPTQConfig, deepspeed
from transformers import Trainer, GPTQConfig, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from transformers import BitsAndBytesConfig

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,  #8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    use_lor2c: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # ['gate_proj', 'o_proj', 'k_proj', 'q_proj', 'up_proj', 'down_proj', 'v_proj']
    lora_target_modules: List[str] = field(
        # default_factory=lambda: ['o_proj', 'k_proj', 'q_proj', 'v_proj']
        default_factory=lambda: ['q_proj', 'v_proj']
    )
    # lora_target_modules = None
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False

@dataclass
class Lor2cArguments:
    # 基本超参
    lor2c_r: int = field(default=8, metadata={"help": "LoR2C rank r"})
    lor2c_alpha: float = field(default=16.0, metadata={"help": "LoR2C alpha (scaling numerator)"})
    lor2c_dropout: float = field(default=0.0, metadata={"help": "Dropout applied on LoR2C residual branch input"})
    lor2c_init_std: float = field(default=0.02, metadata={"help": "Std for A init (B is zero-init by default)"})

    # 作用范围（可选）
    lor2c_target_layers: Optional[str] = field(
        default='all',
        metadata={
            "help": (
                "Which transformer block indices to apply LoR2C. "
                "Examples: 'all', '0,1,2', '0-7', '0-3,8-11'. "
                "Default None means all."
            )
        },
    )

    # 是否训练 LoR2C（有时候只想评估加载）
    lor2c_trainable: bool = field(default=True, metadata={"help": "Whether LoR2C params require_grad=True"})

def parse_layer_indices(spec: Optional[str], n_layers: int):#解析 lor2c_target_layers 的小工具
    if spec is None or spec.strip() == "" or spec.strip().lower() == "all":
        return list(range(n_layers))
    spec = spec.replace(" ", "")
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            for i in range(a, b + 1):
                if 0 <= i < n_layers:
                    out.add(i)
        else:
            i = int(part)
            if 0 <= i < n_layers:
                out.add(i)
    return sorted(out)

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer_deepspeed(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def get_peft_state(named_params, bias: str = "none"):
    """
    从 model.named_parameters() 里筛出要保存的参数。
    - bias="none": 只保存 LoRA 参数
    - bias="all": 保存 LoRA + 所有 bias
    - bias="lora_only": 保存 LoRA + 与 LoRA 对应层的 bias（更省）
    """
    if bias == "none":
        to_return = {k: v for k, v in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: v for k, v in named_params if ("lora_" in k) or (".bias" in k or k.endswith("bias"))}
    elif bias == "lora_only":
        to_return = {}
        lora_bias_names = set()

        # 先找出 LoRA 参数对应的 bias 名称
        for k, v in named_params:
            if "lora_" in k:
                to_return[k] = v
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)

        # 再把这些 bias 加进来
        for k, v in named_params:
            if ("bias" in k) and (k in lora_bias_names):
                to_return[k] = v
    else:
        raise NotImplementedError(f"Unknown bias mode: {bias}")

    # 统一搬到 CPU，避免显存占用 & 保证可保存
    return {k: v.detach().cpu().clone() for k, v in to_return.items()}


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias: str = "none"):
    """
    无 DeepSpeed 的保存逻辑：
    - use_lora: 保存 LoRA state dict（小文件）
    - 否则：保存完整模型
    """
    os.makedirs(output_dir, exist_ok=True)

    if trainer.args.use_lora:
        state_dict = get_peft_state(trainer.model.named_parameters(), bias=bias)
    else:
        state_dict = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}

    # 单卡/主进程保存（HF Trainer 的标准写法）
    if trainer.args.should_save and trainer.args.local_rank in (-1, 0):
        trainer._save(output_dir, state_dict=state_dict)
    


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a pirate chatbot who always responds in pirate speak!"
) -> Dict:

    # im_start = tokenizer.im_start_id
    # im_end = tokenizer.im_end_id

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        input_id, target = [], []
        system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
        input_id += system
        target += [IGNORE_TOKEN_ID] * len(input_id)
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = sentence["from"]
            value = sentence["value"]
            if role == 'user':
                _input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [
                    eot_id]
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            elif role == 'assistant':
                _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [
                    eot_id]
                _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                          [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + tokenizer(value).input_ids + [eot_id]
            else:
                raise NotImplementedError
            input_id += _input_id
            target += _target
        # print(input_id)
        # print(target)
        # print(tokenizer.decode(input_id))
        # print(len(input_id), len(target))
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def build_alpaca_prompt(instruction: str, input_: str = "") -> str:
    instruction = (instruction or "").strip()
    input_ = (input_ or "").strip()

    if input_:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )


def preprocess_alpaca_examples(
    examples,  # list of dicts with keys: instruction, input, output
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict[str, torch.Tensor]:
    """
    For each example:
      prompt = alpaca_template(instruction, input)
      answer = output
      input_ids = tokenize(prompt + answer + eos)
      labels   = IGNORE for prompt tokens, real ids for answer(+eos) tokens
    """
    input_ids_list, labels_list, attn_list = [], [], []

    # Make sure pad exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for ex in examples:
        inst = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")

        prompt = build_alpaca_prompt(inst, inp)
        answer = (out or "").strip()

        # Tokenize prompt and full text separately to get prompt length
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_text = prompt + answer

        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        # Append EOS if not present (important for causal LM SFT)
        if len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id:
            full_ids = full_ids + [tokenizer.eos_token_id]

        # Labels: mask prompt part, supervise answer(+eos) part
        labels = [IGNORE_TOKEN_ID] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # Truncate to max_len
        full_ids = full_ids[:max_len]
        labels = labels[:max_len]

        # Pad to max_len
        pad_len = max_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [IGNORE_TOKEN_ID] * pad_len

        attention_mask = [1 if t != tokenizer.pad_token_id else 0 for t in full_ids]

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attn_list.append(attention_mask)

    return dict(
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        labels=torch.tensor(labels_list, dtype=torch.long),
        attention_mask=torch.tensor(attn_list, dtype=torch.long),
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

class AlpacaSupervisedDataset(Dataset):
    """Alpaca-style SFT dataset: instruction/input/output JSON list."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super().__init__()
        rank0_print("Formatting Alpaca inputs (non-lazy)...")
        data_dict = preprocess_alpaca_examples(raw_data, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class AlpacaLazySupervisedDataset(Dataset):
    """Lazy Alpaca-style SFT dataset."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super().__init__()
        rank0_print("Formatting Alpaca inputs... (lazy mode)")
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cached = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        if i in self.cached:
            return self.cached[i]

        one = preprocess_alpaca_examples([self.raw_data[i]], self.tokenizer, self.max_len)
        item = dict(
            input_ids=one["input_ids"][0],
            labels=one["labels"][0],
            attention_mask=one["attention_mask"][0],
        )
        self.cached[i] = item
        return item


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset_cls = (
    #     LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    # )
    dataset_cls = AlpacaLazySupervisedDataset if data_args.lazy_preprocess else AlpacaSupervisedDataset
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def get_quantization_config(model_args):
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        # if model_args.torch_dtype not in {"auto", None}:
        #     compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config

class SaveLoRAGradCallback(TrainerCallback):
    """
    在每次 optimizer.step() 之前记录 LoRA 梯度（兼容梯度累计/DDP）。
    输出 JSONL：每行一条记录。
    """
    def __init__(self, gradient_log_file, log_every_steps=5,
                 record_stats=True, record_full=False):
        self.gradient_log_file = gradient_log_file
        self.log_every_steps = int(log_every_steps)
        self.record_stats = bool(record_stats)
        self.record_full = bool(record_full)
        self._fp = None

    def on_train_begin(self, args, state, control, **kwargs):
        # 只在主进程写
        if state.is_world_process_zero:
            os.makedirs(os.path.dirname(self.gradient_log_file), exist_ok=True)
            self._fp = open(self.gradient_log_file, "a", encoding="utf-8")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """
        这里触发时：backward 已完成、梯度裁剪也一般已完成（取决于版本），
        但 optimizer.step() 还没做，梯度一定还没被清空。
        """
        if (self._fp is None) or (not state.is_world_process_zero):
            return control

        next_step = int(state.global_step) + 1  # 这一轮将要进行的 optimizer step 编号
        if next_step == 0 or (next_step % self.log_every_steps) != 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            # 某些版本不传 model，可从 trainer 拿
            trainer = kwargs.get("trainer", None)
            model = trainer.model if trainer is not None else None
        if model is None:
            return control

        rec = {
            "time": time.time(),
            "global_step": next_step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "lora": {}
        }

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # PEFT LoRA 常见命名：lora_A / lora_B / lora_*
            if ("lora_" not in name) and ("lora_A" not in name) and ("lora_B" not in name) and ("adapter_weights" not in name):
                continue
            if p.grad is None:
                continue

            g = p.grad.detach()
            item = {}

            if self.record_stats:
                item.update({
                    "shape": list(g.shape),
                    "dtype": str(g.dtype),
                    "norm2": float(torch.norm(g, p=2).item()),
                    "abs_mean": float(g.abs().mean().item()),
                    "abs_max": float(g.abs().max().item()),
                })

            if self.record_full:
                item["full_grad"] = g.float().cpu().tolist()

            rec["lora"][name] = item

        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fp.flush()
        return control

class SaveLoR2CGradCallback(TrainerCallback):
    """
    Log grads BEFORE optimizer.step(), so param.grad is still available.

    - Works for both PEFT LoRA ("lora_") and your LoR2C (contains "lor2c")
    - Writes jsonl:
      {"time":..., "global_step":..., "epoch":..., "grads": {...}, "debug": {...}}
    """

    def __init__(
        self,
        gradient_log_file: str,
        log_every_steps: int = 200,
        record_stats: bool = True,
        record_full: bool = False,
        max_full_elems: int = 4096,
        include_keywords: Optional[List[str]] = None,
        debug: bool = True,
    ):
        self.gradient_log_file = gradient_log_file
        self.log_every_steps = int(log_every_steps)
        self.record_stats = bool(record_stats)
        self.record_full = bool(record_full)
        self.max_full_elems = int(max_full_elems)
        self.debug = bool(debug)

        if include_keywords is None:
            include_keywords = ["lora_", "lor2c"]
        self.include_keywords = [k.lower() for k in include_keywords]

        os.makedirs(os.path.dirname(self.gradient_log_file), exist_ok=True)

    def _want(self, name: str) -> bool:
        n = name.lower()
        return any(k in n for k in self.include_keywords)

    @torch.no_grad()
    def _tensor_stats(self, g: torch.Tensor) -> Dict[str, Any]:
        if g.is_sparse:
            g = g.coalesce().values()
        x = g.detach().float()
        return {
            "shape": list(g.shape),
            "dtype": str(g.dtype),
            "norm2": float(torch.linalg.vector_norm(x).item()),
            "abs_mean": float(x.abs().mean().item()) if x.numel() else 0.0,
            "abs_max": float(x.abs().max().item()) if x.numel() else 0.0,
        }

    @torch.no_grad()
    def _tensor_full(self, g: torch.Tensor) -> Dict[str, Any]:
        if g.is_sparse:
            g = g.coalesce().values()
        flat = g.detach().flatten()
        truncated = False
        if flat.numel() > self.max_full_elems:
            flat = flat[: self.max_full_elems]
            truncated = True
        return {
            "truncated": truncated,
            "numel": int(g.numel()),
            "data": flat.float().cpu().tolist(),
        }

    # ✅关键：在 optimizer.step() 之前记录（此时 grad 还没被清空）
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.log_every_steps <= 0:
            return control
        if state.global_step == 0 or (state.global_step % self.log_every_steps) != 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        grads: Dict[str, Any] = {}
        dbg = {"matched_params": 0, "matched_trainable": 0, "matched_has_grad": 0}

        for name, param in model.named_parameters():
            if not self._want(name):
                continue

            dbg["matched_params"] += 1
            if param.requires_grad:
                dbg["matched_trainable"] += 1
            if param.grad is not None:
                dbg["matched_has_grad"] += 1

            if (not param.requires_grad) or (param.grad is None):
                if not param.requires_grad:
                    print(f"{name}:require_grad {str(param.require_grad)}")
                else:
                    print(f"{name}:grad is None ")
                continue

            entry: Dict[str, Any] = {}
            if self.record_stats:
                entry.update(self._tensor_stats(param.grad))
            if self.record_full:
                entry["full"] = self._tensor_full(param.grad)

            grads[name] = entry

        payload = {
            "time": time.time(),
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "grads": grads,
        }
        if self.debug:
            payload["debug"] = dbg

        try:
            with open(self.gradient_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[SaveAdapterGradCallback] Failed to write grad log: {e}")

        return control


class LoR2CResidual(nn.Module):
    def __init__(self, hidden_size: int, rank: int, alpha: float = 16.0, dropout: float = 0.0, init_std: float = 0.02):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if self.rank > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # B: [d, r], A: [r, d]
        self.B = nn.Parameter(torch.empty(hidden_size, rank))
        self.A = nn.Parameter(torch.empty(rank, hidden_size))

        # init like LoRA: A ~ N(0, std), B = 0
        nn.init.normal_(self.A, mean=0.0, std=init_std)
        nn.init.zeros_(self.B)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
    # ✅保证参数和输入同 device / dtype（只在不一致时移动，避免每步搬运）
        if self.B.device != h.device:
            self.B.data = self.B.data.to(h.device)
        if self.A.device != h.device:
            self.A.data = self.A.data.to(h.device)

        h = self.dropout(h)
        return self.scaling * ((h @ self.B) @ self.A)


# -----------------------------
# Wrapper: out = block(h, ...) + LoR2C(h)
# -----------------------------
class LoR2CBlockWrapper(nn.Module):
    def __init__(self, block: nn.Module, hidden_size: int, rank: int, alpha: float, dropout: float, init_std: float):
        super().__init__()
        self.block = block
        self.lor2c = LoR2CResidual(hidden_size, rank, alpha=alpha, dropout=dropout, init_std=init_std)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.block(hidden_states, *args, **kwargs)

        # 取 block 的输出 hidden_states
        if isinstance(out, tuple):
            base_hs = out[0]
        else:
            base_hs = out

        # PreNorm -> LoR²C
        # if hasattr(self.block, "input_layernorm"):
        #     x = self.block.input_layernorm(hidden_states)
        # else:
        #     x = hidden_states
        # delta = self.lor2c(x)

        delta = self.lor2c(hidden_states)
        new_hs = base_hs + delta

        # 保持 block 的返回结构
        if isinstance(out, tuple):
            return (new_hs,) + out[1:]
        else:
            return new_hs

# -----------------------------
# Utilities to locate blocks + hidden size
# -----------------------------
def infer_hidden_size(model) -> int:
    # Most HF CausalLM configs have hidden_size
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    # fallback
    for name in ["n_embd", "d_model"]:
        if hasattr(model.config, name):
            return int(getattr(model.config, name))
    raise ValueError("Cannot infer hidden size from model.config")

def get_transformer_blocks(model):
    """
    Return (blocks_module_list, setter_fn)
    setter_fn(i, new_block) should replace the i-th block in model.
    """
    # LLaMA / Mistral / Qwen2.* often: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        def setter(i, v): model.model.layers[i] = v
        return layers, setter

    # GPT-like: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        def setter(i, v): model.transformer.h[i] = v
        return layers, setter

    # Some Qwen variants: model.model.blocks (rare)
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        layers = model.model.blocks
        def setter(i, v): model.model.blocks[i] = v
        return layers, setter

    raise ValueError("Unknown model block structure. Add a branch in get_transformer_blocks().")


def inject_lor2c(model, lor2c_args):
    hidden_size = infer_hidden_size(model)
    blocks, setter = get_transformer_blocks(model)

    target_layers = parse_layer_indices(lor2c_args.lor2c_target_layers, n_layers=len(blocks))

    for i in target_layers:
        orig = blocks[i]
        if isinstance(orig, LoR2CBlockWrapper):
            continue
        setter(i, LoR2CBlockWrapper(
            orig,
            hidden_size=hidden_size,
            rank=lor2c_args.lor2c_r,
            alpha=lor2c_args.lor2c_alpha,
            dropout=lor2c_args.lor2c_dropout,
            init_std=lor2c_args.lor2c_init_std,
        ))

    return model



def set_lor2c_trainable(model, trainable: bool = True):
    # 不碰 base 是否训练（你可以自己决定是否全冻结）
    for m in model.modules():
        if isinstance(m, LoR2CResidual):
            for p in m.parameters():
                p.requires_grad = bool(trainable)
    return model


def print_trainable_parameters(model, prefix: str = ""):
    """
    Print the number of trainable parameters in the model, aligned with
    PEFT's `model.print_trainable_parameters()` style.

    Args:
        model: torch.nn.Module
        prefix: optional string prefix for logging (e.g. "[LoR2C]")
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        num = param.numel()
        all_params += num
        if param.requires_grad:
            trainable_params += num

    percent = 100 * trainable_params / all_params if all_params > 0 else 0.0

    header = f"{prefix} " if prefix else ""
    print(
        f"{header}trainable params: {trainable_params:,d} "
        f"|| all params: {all_params:,d} "
        f"|| trainable%: {percent:.4f}"
    )


def freeze_base_model(model):
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze LoR2C params
    for m in model.modules():
        if isinstance(m, LoR2CResidual):
            for p in m.parameters():
                p.requires_grad = True
    return model


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, Lor2cArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        lor2c_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'instruct' in model_args.model_name_or_path.lower()
    # if (
    #         training_args.use_lora
    #         and not lora_args.q_lora
    #         and not is_chat_model
    #         #and deepspeed.is_deepspeed_zero3_enabled() 
    # ):
    #     raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': True,
        # 'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    quantization_config = get_quantization_config(lora_args)

    print("quantization_config：", quantization_config)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config if lora_args.q_lora else None,
        **model_load_kwargs,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -----------------------------
    # Apply LoR2C (before/independent of PEFT LoRA)
    # -----------------------------
    if training_args.use_lor2c:
        grad_cb = SaveLoR2CGradCallback( # record the gradient
            gradient_log_file=os.path.join(training_args.output_dir, "lor2c_grad_stats.jsonl"),
            log_every_steps=50,
            record_stats=True,
            record_full=False,
        )
        model = inject_lor2c(model, lor2c_args)
        print("====== ALL MODULE NAMES ======")
        for name, module in model.named_modules():
            print(name, "->", type(module))
        print("====== END ======")


        # 通常 LoR2C 也是 PEFT：冻结底座，只训 LoR2C（如果你想和 LoRA 一样）
        for p in model.parameters():
            p.requires_grad = False
        set_lor2c_trainable(model, trainable=lor2c_args.lor2c_trainable)
        print_trainable_parameters(model)


    if training_args.use_lora:
        grad_cb = SaveLoRAGradCallback( # record the gradient
            gradient_log_file=os.path.join(training_args.output_dir, "lora_grad_stats.jsonl"),
            log_every_steps=200,
            record_stats=True,
            record_full=False,
        )
        if is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = None
            # modules_to_save = ["wte", "lm_head"]


        def find_all_linear_names(args, model):
            import bitsandbytes as bnb
            cls = bnb.nn.Linear4bit if args.load_in_4bit == 4 else (
                bnb.nn.Linear8bitLt if args.load_in_8bit == 8 else torch.nn.Linear)
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names:  # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)

        if lora_args.lora_target_modules is None:
            lora_args.lora_target_modules = find_all_linear_names(lora_args, model)

        print(lora_args.lora_target_modules)
        print(modules_to_save)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)
        print("====== ALL MODULE NAMES ======")
        for name, module in model.named_modules():
            print(name, "->", type(module))
        print("====== END ======")


        # Print peft trainable params
        # model.print_trainable_parameters()
        print_trainable_parameters(model)


    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner

    callbacks = [grad_cb]

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module
    )

    with torch.autocast("cuda"):
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)
    model.eval()
    prompt = "### Instruction:\nWrite a short greeting.\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    train()
