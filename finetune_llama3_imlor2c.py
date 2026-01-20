# finetune_llama3_lor2c_merge_inject.py

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set, Tuple

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
from transformers import BitsAndBytesConfig
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
logger = logging.getLogger(__name__)

_PRINT_ONCE: Set[str] = set()


def print_once(key: str, msg: str):
    if key in _PRINT_ONCE:
        return
    _PRINT_ONCE.add(key)
    print(msg)


# -----------------------------
# Arguments
# -----------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None, metadata={"help": "Path to training json or jsonl."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to eval json or jsonl."})
    lazy_preprocess: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024, metadata={"help": "Max sequence length."})
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
    lor2c_r: int = field(default=8, metadata={"help": "LoR2C rank"})
    lor2c_alpha: float = field(default=16.0, metadata={"help": "LoR2C alpha"})
    lor2c_dropout: float = field(default=0.0, metadata={"help": "LoR2C dropout"})
    lor2c_init_std: float = field(default=0.02, metadata={"help": "Init std for A"})
    lor2c_target_layers: Optional[str] = field(
        default="all",
        metadata={"help": "Layer indices: 'all', '0,1,2', '0-7', '0-3,8-11'."},
    )
    lor2c_trainable: bool = field(default=True, metadata={"help": "Whether LoR2C params are trainable"})


# -----------------------------
# Common helpers
# -----------------------------
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


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print(*args)


def print_trainable_parameters(model: nn.Module, prefix: str = ""):
    trainable_params = 0
    all_params = 0
    for _, p in model.named_parameters():
        n = p.numel()
        all_params += n
        if p.requires_grad:
            trainable_params += n
    pct = 100.0 * trainable_params / all_params if all_params else 0.0
    head = f"{prefix} " if prefix else ""
    rank0_print(
        f"{head}trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {pct:.4f}"
    )


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

    if getattr(trainer.args, "use_lora", False):
        state_dict = get_peft_state(trainer.model.named_parameters(), bias=bias)
    else:
        state_dict = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}

    if trainer.args.should_save and trainer.args.local_rank in (-1, 0):
        trainer._save(output_dir, state_dict=state_dict)


def load_json_or_jsonl(path: str) -> List[dict]:
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


def preprocess_alpaca_examples(examples: List[dict], tokenizer, max_len: int) -> Dict[str, torch.Tensor]:
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
        d = preprocess_alpaca_examples(raw_data, tokenizer, max_len)
        self.input_ids = d["input_ids"]
        self.labels = d["labels"]
        self.attention_mask = d["attention_mask"]

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


def make_supervised_data_module(tokenizer, data_args: DataArguments, max_len: int) -> Dict:
    dataset_cls = AlpacaLazySupervisedDataset if data_args.lazy_preprocess else AlpacaSupervisedDataset

    if not data_args.data_path:
        raise ValueError("--data_path must be provided for this script.")
    train_rows = load_json_or_jsonl(data_args.data_path)
    train_dataset = dataset_cls(train_rows, tokenizer=tokenizer, max_len=max_len)

    eval_dataset = None
    if data_args.eval_data_path:
        eval_rows = load_json_or_jsonl(data_args.eval_data_path)
        eval_dataset = dataset_cls(eval_rows, tokenizer=tokenizer, max_len=max_len)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# -----------------------------
# LoR2C bridge context
# -----------------------------
class _LoR2CBridgeContext:
    def __init__(self):
        self._inputs: Dict[int, torch.Tensor] = {}

    def reset(self):
        self._inputs = {}

    def push(self, layer_idx: int, hidden_states: torch.Tensor):
        self._inputs[int(layer_idx)] = hidden_states

    def get(self, layer_idx: int) -> torch.Tensor:
        layer_idx = int(layer_idx)
        if layer_idx not in self._inputs:
            raise KeyError(f"Missing input for layer {layer_idx}. Keys={sorted(self._inputs.keys())[:10]}...")
        return self._inputs[layer_idx]


LOR2C_BRIDGE_CTX = _LoR2CBridgeContext()


# -----------------------------
# LoR2C modules
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


class MergeableLoR2CWrapper(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layer_idx: int,
        hidden_size: int,
        rank: int,
        alpha: float,
        dropout: float,
        init_std: float,
    ):
        super().__init__()
        self.block = block
        self.layer_idx = int(layer_idx)

        self.local = LoR2CResidual(hidden_size, rank, alpha=alpha, dropout=dropout, init_std=init_std)
        self.cross = LoR2CResidual(hidden_size, rank, alpha=alpha, dropout=dropout, init_std=init_std)

        for p in self.cross.parameters():
            p.requires_grad = False

        self.use_local = True
        self.use_cross = False
        self.expect_prev_idx: Optional[int] = None

        self.soft_merge_active = False
        self.soft_merge_prev_idx: Optional[int] = None
        self.soft_merge_alpha = 0.0
        self.soft_merge_lower_layer = False

        self.soft_inject_active = False
        self.soft_inject_alpha = 0.0

    def enable_local(self, flag: bool, trainable: bool = True):
        self.use_local = bool(flag)
        for p in self.local.parameters():
            p.requires_grad = bool(trainable) if self.use_local else False

    def enable_cross(self, flag: bool, prev_idx: Optional[int] = None, trainable: bool = True):
        self.use_cross = bool(flag)
        self.expect_prev_idx = int(prev_idx) if prev_idx is not None else None
        for p in self.cross.parameters():
            p.requires_grad = bool(trainable) if self.use_cross else False

    def start_soft_merge(self, prev_idx: int):
        self.soft_merge_active = True
        self.soft_merge_prev_idx = int(prev_idx)
        self.soft_merge_alpha = 0.0
        self.soft_merge_lower_layer = False

    def set_soft_merge_alpha(self, a: float):
        self.soft_merge_alpha = float(max(0.0, min(1.0, a)))

    def finish_soft_merge(self):
        self.soft_merge_active = False
        self.soft_merge_alpha = 1.0

    def start_soft_inject(self):
        self.soft_inject_active = True
        self.soft_inject_alpha = 0.0

    def set_soft_inject_alpha(self, a: float):
        self.soft_inject_alpha = float(max(0.0, min(1.0, a)))

    def finish_soft_inject(self):
        self.soft_inject_active = False
        self.soft_inject_alpha = 1.0

    def forward(self, hidden_states, *args, **kwargs):
        LOR2C_BRIDGE_CTX.push(self.layer_idx, hidden_states)
        out = self.block(hidden_states, *args, **kwargs)
        hs = out[0] if isinstance(out, tuple) else out

        local_add = 0.0
        if self.use_local:
            local_add = self.local(hidden_states)
            if self.soft_inject_active:
                local_add = (1.0 - self.soft_inject_alpha) * local_add
            if self.soft_merge_lower_layer:
                local_add = (1.0 - self.soft_merge_alpha) * local_add

        cross_add = 0.0
        if self.use_cross:
            if self.expect_prev_idx is None:
                raise RuntimeError("use_cross=True but expect_prev_idx is None")
            prev_inp = LOR2C_BRIDGE_CTX.get(self.expect_prev_idx)
            cross_add = self.cross(prev_inp)

        if self.soft_merge_active:
            a = self.soft_merge_alpha
            add = (1.0 - a) * local_add + cross_add
        else:
            add = local_add + cross_add

        hs = hs + add
        return (hs,) + out[1:] if isinstance(out, tuple) else hs


# -----------------------------
# Toggleable LoRA (q/v only)
# -----------------------------
class ToggleLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, enabled: bool = False):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.enabled = bool(enabled)

        self.gate = 1.0

        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f))
        self.lora_B = nn.Parameter(torch.empty(out_f, self.r))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

        self.set_trainable(False)

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)

    def set_trainable(self, flag: bool):
        flag = bool(flag)
        self.lora_A.requires_grad = flag
        self.lora_B.requires_grad = flag

    def set_gate(self, gate: float):
        self.gate = float(max(0.0, min(1.0, gate)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if (not self.enabled) or self.r <= 0:
            return out
        x_d = self.dropout(x)
        mid = x_d.matmul(self.lora_A.t())
        delta = mid.matmul(self.lora_B.t()) * self.scaling
        return out + (self.gate * delta)


# -----------------------------
# Model structure utilities
# -----------------------------
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

    raise ValueError("Unknown model block structure.")


def _unwrap_block(blk: nn.Module) -> nn.Module:
    return blk.block if hasattr(blk, "block") else blk


def _get_qv_from_block(block: nn.Module):
    sa = getattr(block, "self_attn", None)
    if sa is None:
        return None, None
    return getattr(sa, "q_proj", None), getattr(sa, "v_proj", None)


# -----------------------------
# LoR2C inject/merge primitives
# -----------------------------
def inject_lor2c(model, lor2c_args: Lor2cArguments):
    hidden_size = infer_hidden_size(model)
    blocks, setter = get_transformer_blocks(model)
    target_layers = parse_layer_indices(lor2c_args.lor2c_target_layers, n_layers=len(blocks))

    for i in target_layers:
        orig = blocks[i]
        if isinstance(orig, MergeableLoR2CWrapper):
            continue
        setter(
            i,
            MergeableLoR2CWrapper(
                orig,
                layer_idx=i,
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
        if isinstance(m, MergeableLoR2CWrapper):
            m.enable_local(True, trainable=bool(trainable))
            m.enable_cross(False, trainable=False)
    return model


def preinstall_qv_lora_all_layers(model, r_half: int, alpha: float, dropout: float):
    blocks, _ = get_transformer_blocks(model)
    for blk in blocks:
        base = _unwrap_block(blk)
        q, v = _get_qv_from_block(base)

        if isinstance(q, nn.Linear) and (not isinstance(q, ToggleLoRALinear)):
            base.self_attn.q_proj = ToggleLoRALinear(q, r=r_half, alpha=alpha, dropout=dropout, enabled=False)
        if isinstance(v, nn.Linear) and (not isinstance(v, ToggleLoRALinear)):
            base.self_attn.v_proj = ToggleLoRALinear(v, r=r_half, alpha=alpha, dropout=dropout, enabled=False)


def enable_qv_lora_for_layer(model, layer_idx: int, trainable: bool = True):
    blocks, _ = get_transformer_blocks(model)
    blk = blocks[layer_idx]
    base = _unwrap_block(blk)
    q, v = _get_qv_from_block(base)

    if isinstance(q, ToggleLoRALinear):
        q.set_enabled(True)
        q.set_trainable(trainable)
    if isinstance(v, ToggleLoRALinear):
        v.set_enabled(True)
        v.set_trainable(trainable)


def set_qv_lora_gate_for_layer(model, layer_idx: int, gate: float):
    blocks, _ = get_transformer_blocks(model)
    blk = blocks[layer_idx]
    base = _unwrap_block(blk)
    q, v = _get_qv_from_block(base)

    if isinstance(q, ToggleLoRALinear):
        q.set_gate(gate)
    if isinstance(v, ToggleLoRALinear):
        v.set_gate(gate)


@torch.no_grad()
def stable_rank_from_lor2c_AB(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> float:
    A = A.detach().to(dtype=torch.float32).cpu().clone()
    B = B.detach().to(dtype=torch.float32).cpu().clone()

    G1 = B.T @ B
    G2 = A @ A.T
    M = G1 @ G2

    eigvals = torch.linalg.eigvals(M).real
    eigvals = torch.clamp(eigvals, min=0.0)

    fro2 = float(eigvals.sum().item())
    spec2 = float(eigvals.max().item()) if eigvals.numel() else 0.0

    if spec2 <= eps:
        return 0.0
    return float(fro2 / (spec2 + eps))


@torch.no_grad()
def _lowrank_svd_factors_from_AB(A: torch.Tensor, B: torch.Tensor):
    A = A.detach().to(dtype=torch.float32).cpu().clone()
    B = B.detach().to(dtype=torch.float32).cpu().clone()

    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")

    Mmid = Rb @ Ra.T
    Um, Sm, VmT = torch.linalg.svd(Mmid, full_matrices=False)
    Vm = VmT.T
    Sm = torch.clamp(Sm, min=0.0)
    return Qb, Qa, Um, Sm, Vm, Mmid


@torch.no_grad()
def symalign_from_lor2c_AB(A1: torch.Tensor, B1: torch.Tensor, A2: torch.Tensor, B2: torch.Tensor, eps: float = 1e-12) -> float:
    Qb1, Qa1, Um1, S1, Vm1, M1 = _lowrank_svd_factors_from_AB(A1, B1)
    Qb2, Qa2, Um2, S2, Vm2, M2 = _lowrank_svd_factors_from_AB(A2, B2)

    w1_fro2 = float((S1 * S1).sum().item())
    w2_fro2 = float((S2 * S2).sum().item())
    if w1_fro2 <= eps or w2_fro2 <= eps:
        return 0.0

    Sb = Qb2.T @ Qb1
    Sa = Qa1.T @ Qa2

    X12 = (Um2.T @ Sb @ M1 @ Sa @ Vm2)
    term1 = float((X12 * X12).sum().item()) / (w1_fro2 + eps)

    Sb2 = Qb1.T @ Qb2
    Sa2 = Qa2.T @ Qa1
    X21 = (Um1.T @ Sb2 @ M2 @ Sa2 @ Vm1)
    term2 = float((X21 * X21).sum().item()) / (w2_fro2 + eps)

    out = 0.5 * (term1 + term2)
    return float(max(0.0, min(1.0, out)))


@torch.no_grad()
def pick_min_stablerank_for_inject(model, exclude_layers: Optional[Set[int]] = None):
    exclude_layers = exclude_layers or set()
    blocks, _ = get_transformer_blocks(model)

    best: Optional[Tuple[float, int]] = None
    for i, blk in enumerate(blocks):
        if i in exclude_layers:
            continue
        if not isinstance(blk, MergeableLoR2CWrapper):
            continue
        if (not blk.use_local) or blk.use_cross:
            continue

        sr = float(stable_rank_from_lor2c_AB(blk.local.A, blk.local.B))
        if best is None or sr < best[0]:
            best = (sr, i)
    return best


@torch.no_grad()
def pick_max_symalign_adjacent_pair(model, eps: float = 1e-12):
    blocks, _ = get_transformer_blocks(model)
    n = len(blocks)

    best: Optional[Tuple[float, int, int]] = None
    for t in range(n - 1):
        b0, b1 = blocks[t], blocks[t + 1]
        if not (isinstance(b0, MergeableLoR2CWrapper) and isinstance(b1, MergeableLoR2CWrapper)):
            continue
        if (not b0.use_local) or (not b1.use_local) or b1.use_cross:
            continue

        sym = float(symalign_from_lor2c_AB(b0.local.A, b0.local.B, b1.local.A, b1.local.B, eps=eps))
        if best is None or sym > best[0]:
            best = (sym, t, t + 1)

    if best is None:
        return None
    sym, t, tp1 = best
    return (t, tp1, sym)


@torch.no_grad()
def start_soft_merge_adjacent_pair(model, t: int, tp1: int):
    blocks, _ = get_transformer_blocks(model)
    bt = blocks[t]
    b1 = blocks[tp1]

    if not (isinstance(bt, MergeableLoR2CWrapper) and isinstance(b1, MergeableLoR2CWrapper)):
        raise ValueError("Blocks must be MergeableLoR2CWrapper for merging.")
    if tp1 != t + 1:
        raise ValueError("Only adjacent pairs (t, t+1) are supported.")

    b1.enable_cross(True, prev_idx=t, trainable=True)
    bt.soft_merge_lower_layer = True
    b1.start_soft_merge(prev_idx=t)

    return {"merged": True, "mode": "soft", "pair": [int(t), int(tp1)]}


@torch.no_grad()
def start_soft_inject_layer(model, layer_idx: int):
    blocks, _ = get_transformer_blocks(model)
    blk = blocks[layer_idx]
    if not isinstance(blk, MergeableLoR2CWrapper):
        raise ValueError(f"Layer {layer_idx} is not MergeableLoR2CWrapper")

    enable_qv_lora_for_layer(model, layer_idx, trainable=True)
    set_qv_lora_gate_for_layer(model, layer_idx, gate=0.0)

    blk.start_soft_inject()
    return {"injected": True, "mode": "soft", "layer": int(layer_idx)}


# -----------------------------
# Callbacks
# -----------------------------
class MergeLoR2CCallback(TrainerCallback):
    def __init__(
        self,
        lor2c_args: Lor2cArguments,
        merge_every_steps: int = 200,
        max_merges: int = 999,
        topk: int = 4,
        gate_warmup_steps: int = 300,
        hard_switch: bool = True,
        log_file: Optional[str] = None,
    ):
        self.lor2c_args = lor2c_args
        self.merge_every_steps = int(merge_every_steps)
        self.max_merges = int(max_merges)
        self.topk = int(topk)
        self.gate_warmup_steps = int(gate_warmup_steps)
        self.hard_switch = bool(hard_switch)

        self.merged_times = 0
        self.merged_pairs: List[List[int]] = []
        self._active_pair: Optional[Dict[str, int]] = None

        self.log_file = log_file
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def _log(self, payload: dict):
        if not self.log_file:
            return
        payload = dict(payload)
        payload.update({"time": time.time()})
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _update_gate(self, model, step: int):
        if self._active_pair is None:
            return
        t = self._active_pair["t"]
        tp1 = self._active_pair["tp1"]
        start = self._active_pair["start_step"]

        blocks, _ = get_transformer_blocks(model)
        b1 = blocks[tp1]
        if not isinstance(b1, MergeableLoR2CWrapper):
            self._active_pair = None
            return

        prog = (step - start) / max(1, self.gate_warmup_steps)
        a = float(max(0.0, min(1.0, prog)))
        b1.set_soft_merge_alpha(a)

        if a >= 1.0 - 1e-9:
            if self.hard_switch:
                bt = blocks[t]
                if isinstance(bt, MergeableLoR2CWrapper):
                    bt.enable_local(False, trainable=False)
                b1.enable_local(False, trainable=False)
                b1.finish_soft_merge()
            self._active_pair = None

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control

        step = int(state.global_step)
        self._update_gate(model, step)

        if self._active_pair is not None:
            return control
        if self.merged_times >= self.max_merges:
            return control
        if self.merge_every_steps <= 0:
            return control
        if step == 0 or (step % self.merge_every_steps) != 0:
            return control

        best = pick_max_symalign_adjacent_pair(model)
        if best is None:
            return control
        t, tp1, sym = best

        info = start_soft_merge_adjacent_pair(model, t, tp1)

        optimizer = kwargs.get("optimizer", None)
        if optimizer is None:
            trainer = kwargs.get("trainer", None)
            optimizer = getattr(trainer, "optimizer", None) if trainer is not None else None

        if optimizer is not None:
            blocks, _ = get_transformer_blocks(model)
            b1 = blocks[tp1]
            if isinstance(b1, MergeableLoR2CWrapper) and getattr(b1, "cross", None) is not None:
                for p in b1.cross.parameters():
                    p.requires_grad = True
                opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
                missing = [p for p in b1.cross.parameters() if id(p) not in opt_param_ids]
                if missing:
                    optimizer.add_param_group({"params": missing})

        self.merged_times += 1
        self._active_pair = {"t": int(t), "tp1": int(tp1), "start_step": step}
        self.merged_pairs.append([int(t), int(tp1)])

        self._log(
            dict(
                info,
                global_step=step,
                epoch=float(state.epoch) if state.epoch is not None else None,
                symalign=float(sym),
                merged_times=int(self.merged_times),
                gate_warmup_steps=int(self.gate_warmup_steps),
            )
        )
        return control


class InjectLoR2CCallback(TrainerCallback):
    def __init__(
        self,
        lor2c_args: Lor2cArguments,
        inject_start_step: int = 0,
        inject_every_steps: int = 200,
        max_injects: int = 999,
        topk: int = 4,
        inject_warmup_steps: int = 300,
        hard_switch: bool = True,
        log_file: Optional[str] = None,
    ):
        self.lor2c_args = lor2c_args
        self.inject_start_step = int(inject_start_step)
        self.inject_every_steps = int(inject_every_steps)
        self.max_injects = int(max_injects)
        self.topk = int(topk)
        self.inject_warmup_steps = int(inject_warmup_steps)
        self.hard_switch = bool(hard_switch)

        self.injected_times = 0
        self.injected_layers: Set[int] = set()
        self._active: Optional[Dict[str, int]] = None

        self.log_file = log_file
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def _log(self, payload: dict):
        if not self.log_file:
            return
        payload = dict(payload)
        payload.update({"time": time.time()})
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _update_gate(self, model, step: int):
        if self._active is None:
            return
        idx = self._active["layer"]
        start = self._active["start_step"]

        blocks, _ = get_transformer_blocks(model)
        blk = blocks[idx]
        if not isinstance(blk, MergeableLoR2CWrapper):
            self._active = None
            return

        prog = (step - start) / max(1, self.inject_warmup_steps)
        a = float(max(0.0, min(1.0, prog)))

        blk.set_soft_inject_alpha(a)
        set_qv_lora_gate_for_layer(model, idx, gate=a)

        if a >= 1.0 - 1e-9:
            if self.hard_switch:
                blk.enable_local(False, trainable=False)
                blk.finish_soft_inject()
                set_qv_lora_gate_for_layer(model, idx, gate=1.0)
            self._active = None

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control

        step = int(state.global_step)
        self._update_gate(model, step)

        if self._active is not None:
            return control
        if self.injected_times >= self.max_injects:
            return control
        if step < self.inject_start_step:
            return control
        if self.inject_every_steps <= 0:
            return control
        if (step == self.inject_start_step) or ((step - self.inject_start_step) % self.inject_every_steps != 0):
            return control

        best = pick_min_stablerank_for_inject(model, exclude_layers=self.injected_layers)
        if best is None:
            return control
        srank, idx = best

        info = start_soft_inject_layer(model, idx)

        optimizer = kwargs.get("optimizer", None)
        if optimizer is None:
            trainer = kwargs.get("trainer", None)
            optimizer = getattr(trainer, "optimizer", None) if trainer is not None else None

        if optimizer is not None:
            opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
            missing = []
            key_layer = f"layers.{int(idx)}."

            for name, p in model.named_parameters():
                if key_layer not in name:
                    continue
                if p.requires_grad and (id(p) not in opt_param_ids):
                    missing.append(p)

            if missing:
                uniq = []
                seen = set()
                for p in missing:
                    pid = id(p)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    uniq.append(p)
                optimizer.add_param_group({"params": uniq})

        self.injected_layers.add(int(idx))
        self.injected_times += 1
        self._active = {"layer": int(idx), "start_step": step}

        self._log(
            dict(
                info,
                global_step=step,
                epoch=float(state.epoch) if state.epoch is not None else None,
                stable_rank=float(srank),
                injected_times=int(self.injected_times),
                inject_warmup_steps=int(self.inject_warmup_steps),
            )
        )
        return control


# -----------------------------
# Training entry
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
        low_cpu_mem_usage=True,
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

    callbacks: List[TrainerCallback] = []

    if training_args.use_lor2c:
        merge_cb = MergeLoR2CCallback(
            lor2c_args=lor2c_args,
            merge_every_steps=1200,
            max_merges=8,
            topk=4,
            gate_warmup_steps=598,
            hard_switch=True,
            log_file=os.path.join(training_args.output_dir, "merge_log.jsonl"),
        )
        inject_cb = InjectLoR2CCallback(
            lor2c_args=lor2c_args,
            inject_start_step=600,
            inject_every_steps=1200,
            max_injects=8,
            topk=4,
            inject_warmup_steps=598,
            hard_switch=True,
            log_file=os.path.join(training_args.output_dir, "inject_log.jsonl"),
        )

        model = inject_lor2c(model, lor2c_args)

        r_half = max(1, int(lor2c_args.lor2c_r) // 2)
        preinstall_qv_lora_all_layers(
            model,
            r_half=r_half,
            alpha=float(lor2c_args.lor2c_alpha),
            dropout=float(lor2c_args.lor2c_dropout),
        )

        for p in model.parameters():
            p.requires_grad = False
        set_lor2c_trainable(model, trainable=bool(lor2c_args.lor2c_trainable))
        print_trainable_parameters(model, prefix="[LoR2C]")

        callbacks.extend([merge_cb, inject_cb])

    if training_args.use_lora:
        lora_target = lora_args.lora_target_modules or ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_target,
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

    data_module = make_supervised_data_module(tokenizer, data_args, training_args.model_max_length)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
