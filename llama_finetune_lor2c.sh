#!/bin/bash
export WANDB_MODE=offline
gpu=0,1,2,3

run(){
  bs=128
  micro_bs=1
  learning_rate='3e-4'
  num_train_epochs=3
  mode=$1
  lora_r=$3
  lor2c_r=$2
  max_merge_count=$4
  max_distribution_count=$5
  sfs_k=$6
  seed=42
  lora_alpha="16"
  lor2c_alpha="32"
  target_name='qv'
  lora_dropout=0.05
  lora_bias=none
  cutoff_len=256
  wandb_project=proejct_name
  wandb_run_name=llama-lor2c-${target_name}-${mode}-lora_r-${lora_r}-lor2c_r-${lor2c_r}-n-${l_num}-alpha-16-seed-${seed}-bs-${bs}-lr-${learning_rate}-len-${cutoff_len}-epochs-${num_train_epochs}-merge-${max_merge_count}-dist-${max_distribution_count}
  echo $wandb_run_name
  exp_dir=../llama-lor2c/${wandb_run_name}
  mkdir -p $exp_dir
  
  CUDA_VISIBLE_DEVICES=$gpu python llama_finetune_lor2c.py \
    --base_model= meta-llama/Llama-2-7b-hf \
    --cutoff_len=$cutoff_len \
    --mode=$mode \
    --seed=$seed \
    --group_by_length \
    --lora_r=$lora_r \
    --lor2c_r=$lor2c_r \
    --lora_n=$l_num \
    --lora_alpha=$lora_alpha \
    --lor2c_alpha=$lor2c_alpha \
    --lora_dropout=$lora_dropout \
    --lora_target_modules='[q_proj,v_proj]' \
    --batch_size=$bs \
    --micro_batch_size=$micro_bs \
    --num_epochs=$num_train_epochs \
    --learning_rate=$learning_rate \
    --wandb_project=$wandb_project \
    --wandb_run_name=$wandb_run_name \
    --output_dir=${exp_dir}/model \
    --overwrite_output_dir \
    --max_merge_count ${max_merge_count} \
    --max_distribution_count ${max_distribution_count} \
    --sfs_k ${sfs_k}
}

run 'lor2c' 64 32 8 8
