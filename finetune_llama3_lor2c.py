# finetune_llama3_lor2c.py

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from transformers import BitsAndBytesConfig


logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# -----------------------------
# Arguments
# -----------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    # Backward-compatible: local file path (json/jsonl) or HF dataset id
    data_path: Optional[str] = field(default=None, metadata={"help": "Local path or HF dataset id."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Local path or HF dataset id."})

    # Preferred explicit HF dataset fields
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HF dataset id (e.g., tatsu-lab/alpaca)."})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "HF dataset config name (if any)."})
    dataset_split: str = field(default="train", metadata={"help": "HF dataset split."})

    eval_dataset_name: Optional[str] = field(default=None, metadata={"help": "HF dataset id for eval."})
    eval_dataset_config: Optional[str] = field(default=None, metadata={"help": "HF dataset config name (if any)."})
    eval_dataset_split: str = field(default="validation", metadata={"help": "HF dataset eval split."})

    lazy_preprocess: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Max sequence length (right padded/truncated)."},
    )
    use_lora: bool = field(default=False)
    use_lor2c: bool = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_weight_path: str = ""
    lora_bias: str = "none"

    q_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class Lor2cArguments:
    lor2c_r: int = field(default=8, metadata={"help": "LoR2C rank r"})
    lor2c_alpha: float = field(default=16.0, metadata={"help": "LoR2C alpha"})
    lor2c_dropout: float = field(default=0.0, metadata={"help": "Dropout on LoR2C branch input"})
    lor2c_init_std: float = field(default=0.02, metadata={"help": "Std for A init (B is zero-init)"})

    lor2c_target_layers: Optional[str] = field(
        default="all",
        metadata={"help": "Which block indices to apply LoR2C: 'all', '0,1,2', '0-7', '0-3,8-11'."},
    )
    lor2c_trainable: bool = field(default=True, metadata={"help": "Whether LoR2C params are trainable"})


# -----------------------------
# Helpers
# -----------------------------
def rank0_print(*args):
    lr = int(os.environ.get("LOCAL_RANK", "-1"))
    if lr in (-1, 0):
        print(*args)


def parse_layer_indices(spec: Optional[str], n_layers: int) -> List[int]:
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


def get_peft_state(named_params, bias: str = "none") -> Dict[str, torch.Tensor]:
    if bias == "none":
        to_return = {k: v for k, v in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: v for k, v in named_params if ("lora_" in k) or (".bias" in k) or k.endswith("bias")}
    elif bias == "lora_only":
        to_return = {}
        lora_bias_names = set()
        for k, v in named_params:
            if "lora_" in k:
                to_return[k] = v
                lora_bias_names.add(k.split("lora_")[0] + "bias")
        for k, v in named_params:
            if ("bias" in k) and (k in lora_bias_names):
                to_return[k] = v
    else:
        raise NotImplementedError(f"Unknown bias mode: {bias}")

    return {k: v.detach().cpu().clone() for k, v in to_return.items()}


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str, bias: str = "none"):
    os.makedirs(output_dir, exist_ok=True)
    if trainer.args.use_lora:
        state_dict = get_peft_state(trainer.model.named_parameters(), bias=bias)
    else:
        state_dict = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}

    if trainer.args.should_save and trainer.args.local_rank in (-1, 0):
        trainer._save(output_dir, state_dict=state_dict)


def get_quantization_config(lora_args: LoraArguments):
    if lora_args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    if lora_args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# -----------------------------
# Alpaca SFT formatting
# -----------------------------
def build_alpaca_prompt(instruction: str, input_: str = "") -> str:
    instruction = (instruction or "").strip()
    input_ = (input_ or "").strip()
    if input_:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def preprocess_alpaca_examples(
    examples: List[dict],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids_list, labels_list, attn_list = [], [], []

    for ex in examples:
        inst = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")

        prompt = build_alpaca_prompt(inst, inp)
        answer = (out or "").strip()

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(prompt + answer, add_special_tokens=False).input_ids

        if not full_ids or full_ids[-1] != tokenizer.eos_token_id:
            full_ids = full_ids + [tokenizer.eos_token_id]

        labels = [IGNORE_TOKEN_ID] * len(prompt_ids) + full_ids[len(prompt_ids):]

        full_ids = full_ids[:max_len]
        labels = labels[:max_len]

        pad_len = max_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [IGNORE_TOKEN_ID] * pad_len

        attention_mask = [0 if t == tokenizer.pad_token_id else 1 for t in full_ids]

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attn_list.append(attention_mask)

    return dict(
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        labels=torch.tensor(labels_list, dtype=torch.long),
        attention_mask=torch.tensor(attn_list, dtype=torch.long),
    )


class AlpacaSupervisedDataset(Dataset):
    def __init__(self, raw_data: List[dict], tokenizer, max_len: int):
        super().__init__()
        rank0_print("Building dataset (non-lazy)...")
        data = preprocess_alpaca_examples(raw_data, tokenizer, max_len)
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]
        self.attention_mask = data["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class AlpacaLazySupervisedDataset(Dataset):
    def __init__(self, raw_data: List[dict], tokenizer, max_len: int):
        super().__init__()
        rank0_print("Building dataset (lazy)...")
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache: Dict[int, dict] = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        if i in self.cache:
            return self.cache[i]
        one = preprocess_alpaca_examples([self.raw_data[i]], self.tokenizer, self.max_len)
        item = dict(
            input_ids=one["input_ids"][0],
            labels=one["labels"][0],
            attention_mask=one["attention_mask"][0],
        )
        self.cache[i] = item
        return item


def _load_local_json_or_jsonl(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_hf_dataset_as_list(name: str, config: Optional[str], split: str) -> List[dict]:
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    cols = set(ds.column_names)

    def map_row(x: dict) -> dict:
        if {"instruction", "input", "output"}.issubset(cols):
            return {"instruction": x.get("instruction", ""), "input": x.get("input", ""), "output": x.get("output", "")}
        if {"prompt", "completion"}.issubset(cols):
            return {"instruction": x.get("prompt", ""), "input": "", "output": x.get("completion", "")}
        if "text" in cols:
            return {"instruction": x.get("text", ""), "input": "", "output": ""}
        raise ValueError(f"Unsupported dataset schema: {ds.column_names}")

    return [map_row(r) for r in ds]


def _resolve_train_data(data_args: DataArguments) -> List[dict]:
    if data_args.dataset_name:
        return _load_hf_dataset_as_list(data_args.dataset_name, data_args.dataset_config, data_args.dataset_split)

    if not data_args.data_path:
        raise ValueError("Provide either --dataset_name or --data_path.")

    if os.path.exists(data_args.data_path):
        return _load_local_json_or_jsonl(data_args.data_path)

    return _load_hf_dataset_as_list(data_args.data_path, None, data_args.dataset_split)


def _resolve_eval_data(data_args: DataArguments) -> Optional[List[dict]]:
    if data_args.eval_dataset_name:
        return _load_hf_dataset_as_list(data_args.eval_dataset_name, data_args.eval_dataset_config, data_args.eval_dataset_split)

    if not data_args.eval_data_path:
        return None

    if os.path.exists(data_args.eval_data_path):
        return _load_local_json_or_jsonl(data_args.eval_data_path)

    return _load_hf_dataset_as_list(data_args.eval_data_path, None, data_args.eval_dataset_split)


def make_supervised_data_module(tokenizer, data_args: DataArguments, max_len: int) -> Dict:
    dataset_cls = AlpacaLazySupervisedDataset if data_args.lazy_preprocess else AlpacaSupervisedDataset

    rank0_print("Loading training data...")
    train_rows = _resolve_train_data(data_args)
    train_dataset = dataset_cls(train_rows, tokenizer=tokenizer, max_len=max_len)

    eval_rows = _resolve_eval_data(data_args)
    eval_dataset = dataset_cls(eval_rows, tokenizer=tokenizer, max_len=max_len) if eval_rows else None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# -----------------------------
# LoR2C
# -----------------------------
class LoR2CResidual(nn.Module):
    def __init__(self, hidden_size: int, rank: int, alpha: float, dropout: float, init_std: float):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if self.rank > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.B = nn.Parameter(torch.empty(hidden_size, self.rank))
        self.A = nn.Parameter(torch.empty(self.rank, hidden_size))

        nn.init.normal_(self.A, mean=0.0, std=init_std)
        nn.init.zeros_(self.B)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.dropout(h)
        return self.scaling * ((h @ self.B) @ self.A)


class LoR2CBlockWrapper(nn.Module):
    def __init__(self, block: nn.Module, hidden_size: int, rank: int, alpha: float, dropout: float, init_std: float):
        super().__init__()
        self.block = block
        self.lor2c = LoR2CResidual(hidden_size, rank, alpha=alpha, dropout=dropout, init_std=init_std)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.block(hidden_states, *args, **kwargs)
        base_hs = out[0] if isinstance(out, tuple) else out
        delta = self.lor2c(hidden_states)
        new_hs = base_hs + delta
        return (new_hs,) + out[1:] if isinstance(out, tuple) else new_hs


def infer_hidden_size(model) -> int:
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    for name in ("n_embd", "d_model"):
        if hasattr(model.config, name):
            return int(getattr(model.config, name))
    raise ValueError("Cannot infer hidden size from model.config")


def get_transformer_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        def setter(i, v): model.model.layers[i] = v
        return layers, setter

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        def setter(i, v): model.transformer.h[i] = v
        return layers, setter

    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        layers = model.model.blocks
        def setter(i, v): model.model.blocks[i] = v
        return layers, setter

    raise ValueError("Unknown transformer block structure.")


def inject_lor2c(model, lor2c_args: Lor2cArguments):
    hidden_size = infer_hidden_size(model)
    blocks, setter = get_transformer_blocks(model)
    target_layers = parse_layer_indices(lor2c_args.lor2c_target_layers, n_layers=len(blocks))

    for i in target_layers:
        orig = blocks[i]
        if isinstance(orig, LoR2CBlockWrapper):
            continue
        setter(
            i,
            LoR2CBlockWrapper(
                orig,
                hidden_size=hidden_size,
                rank=lor2c_args.lor2c_r,
                alpha=lor2c_args.lor2c_alpha,
                dropout=lor2c_args.lor2c_dropout,
                init_std=lor2c_args.lor2c_init_std,
            ),
        )
    return model


def set_lor2c_trainable(model, trainable: bool = True):
    for m in model.modules():
        if isinstance(m, LoR2CResidual):
            for p in m.parameters():
                p.requires_grad = bool(trainable)
    return model


def print_trainable_parameters(model, prefix: str = ""):
    trainable_params = 0
    all_params = 0
    for _, p in model.named_parameters():
        n = p.numel()
        all_params += n
        if p.requires_grad:
            trainable_params += n
    pct = 100.0 * trainable_params / all_params if all_params else 0.0
    head = f"{prefix} " if prefix else ""
    rank0_print(f"{head}trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {pct:.4f}")


# -----------------------------
# Train
# -----------------------------
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments, Lor2cArguments))
    model_args, data_args, training_args, lora_args, lor2c_args = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"

    model_load_kwargs = {"low_cpu_mem_usage": True}

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    quantization_config = get_quantization_config(lora_args)

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

    if training_args.use_lor2c:
        model = inject_lor2c(model, lor2c_args)
        for p in model.parameters():
            p.requires_grad = False
        set_lor2c_trainable(model, trainable=lor2c_args.lor2c_trainable)
        print_trainable_parameters(model, prefix="[LoR2C]")

    if training_args.use_lora:
        lora_target_modules = lora_args.lora_target_modules

        if lora_target_modules is None:
            try:
                import bitsandbytes as bnb
                cls = bnb.nn.Linear4bit if lora_args.load_in_4bit else (bnb.nn.Linear8bitLt if lora_args.load_in_8bit else torch.nn.Linear)
            except Exception:
                cls = torch.nn.Linear

            names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    parts = name.split(".")
                    names.add(parts[-1])
            names.discard("lm_head")
            lora_target_modules = sorted(list(names))

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=None,
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model, prefix="[LoRA]")

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    with torch.autocast(device_type="cuda"):
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
