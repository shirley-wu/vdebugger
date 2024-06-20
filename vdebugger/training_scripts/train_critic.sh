MODEL_SIZE=7b
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
if [ $NUM_GPUS != "$( python -c 'import torch; print(torch.cuda.device_count())' )" ]; then
  echo "Hasn't set gpu right"
  exit
fi
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 23466 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path codellama/CodeLlama-${MODEL_SIZE}-Python-hf \
    --use_flash_attn \
    --gradient_checkpointing \
    --tokenizer_name codellama/CodeLlama-${MODEL_SIZE}-Python-hf \
    --use_slow_tokenizer \
    --train_file CRITIQUE_DATA/train-mix.json \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir outputs/critique_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1