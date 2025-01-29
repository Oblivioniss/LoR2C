#!/bin/bash


declare -A epochs=(["mnli"]=30 ["mrpc"]=30 ["qnli"]=30 ["qqp"]=30 ["rte"]=80 ["sst2"]=60 ["stsb"]=40 ["cola"]=30)
declare -A bs=(["mnli"]=128 ["mrpc"]=64 ["qnli"]=64 ["qqp"]=32 ["rte"]=32 ["sst2"]=32 ["stsb"]=32 ["cola"]=16)
declare -A ml=(["mnli"]=256 ["mrpc"]=256 ["qnli"]=256 ["qqp"]=256 ["rte"]=512 ["sst2"]=256 ["stsb"]=256 ["cola"]=256)
declare -A lr=(["mnli"]="5e-4" ["mrpc"]="4e-4" ["qnli"]="4e-4" ["qqp"]="4e-4" ["rte"]="4e-4" ["sst2"]="5e-4" ["stsb"]="4e-4" ["cola"]="4e-4")
declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

export WANDB_MODE=offline
# RTE    2, 490; 277 (5)
# MRPC   3, 668; 408 (8)
# STSB   5, 749; 1, 379 (11)
# CoLA   8, 551; 1, 043 (17)
# QNLI   9, 815; 9, 832 (19)
# SST2  67, 350; 873 (132)
# QQP  363, 870; 40, 431 (711)
# MNLI 392, 702; 9, 815 9, 832 (768)

run(){
  task_name=$1
  learning_rate=${lr[$1]}
  num_train_epochs=${epochs[$1]}
  per_device_train_batch_size=${bs[$1]}
  lora_rank=$3
  lor2c_rank=$2
  seed=17
  lora_alpha="16"
  lor2c_alpha="16"
  target_modules="query value"
  mode=$4
  lora_dropout=0.05
  lora_bias=none
  lora_task_type=SEQ_CLS
  wandb_project=project_name
  share=false
  share_lor2c=false
  local_test=false
  sfs=$7
  max_merge_count=$5
  max_distribution_count=$6
  wandb_run_name=roberta-mhvbsfs-${mode}-${task_name}-r-${lora_rank}-n-${l_num}-alpha-16-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}-merge-${max_merge_count}-dist-${max_distribution_count}
  
  exp_dir=../roberta_glue_reproduce/${wandb_run_name}
  mkdir -p $exp_dir
  export DATASET_ROOT=/data/zhaojiancheng-slurm/project/MSLoRA/data/
  export METRICS_ROOT=/data/zhaojiancheng-slurm/project/MSLoRA/metrics
  cp -r ./peft-0.5.0 $exp_dir
  cp glue_finetune.sh $exp_dir
  cp ./run_glue_lora.py $exp_dir

  
  CUDA_VISIBLE_DEVICES=0 python ./run_glue_lor2c.py \
  --model_name_or_path ./models/roberta-base  \
  --task_name ${task_name} \
  --do_train --do_eval \
  --max_seq_length ${ml[$1]} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end True --metric_for_best_model ${metrics[$1]} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} --wandb_project ${wandb_project} \
  --lora_alpha ${lora_alpha} --lor2c_alpha ${lor2c_alpha} --lora_dropout ${lora_dropout} --lora_bias ${lora_bias} \
  --lora_task_type ${lora_task_type} --target_modules ${target_modules} --rank ${lora_rank} --lor2c_rank ${lor2c_rank} \
  --mode ${mode} \
  --max_merge_count ${max_merge_count} \
  --max_distribution_count ${max_distribution_count} \
  --sfs_k ${sfs} \
  --share_lor2c ${share_lor2c} \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --run_name ${wandb_run_name} \
  --overwrite_output_dir \
  --local_test ${local_test}
  #--max_train_samples 128 \
  #--max_eval_samples 32 \
  #--max_predict_samples 3
}
task_base=('mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'cola')
#for task in "${task_base[@]}"; do
#    run $task "8" "4" "lor2c" "0" "0" "2"
#done
run "mrpc" "8" "4" "lor2c" "0" "0" "2"