LR=2e-5
GRAD_STEPS=1
TRAIN_BS=512
VALID_BS=128
EPOCHS=6
WARMUP_STEPS=500
MAX_STEPS=-1
GPUS=0,1,2,3,4,5,6,7
GPU_COUNT=8
NUM_WORKS=64

TEMP=0.05
MARGIN=0.5
DROPOUT_P=0.1

## Augmented data w/ deduplication
DATA_DIR=../Processed_Data
TRAIN_DIR="${DATA_DIR}/train"
VALID_DIR="${DATA_DIR}/valid"

options=(
   '--loss MLE_Only'
   '--loss ContraCLMSeq --temperature $TEMP'
   '--loss ContraCLMTok --temperature $TEMP'
   '--loss ContraCLM --temperature $TEMP'
)

CL_Config=$(eval echo ${options[3]})

CUDA_VISIBLE_DEVICES=$GPUS python ../pl_trainer.py \
    --num_workers $NUM_WORKS \
    --devices $GPU_COUNT \
    --accelerator gpu \
    --model_name google/gemma-7b \
    --pad_token_id 0 \
    --dropout_p $DROPOUT_P \
    --expt_prefix ccltext_sentence \
    --default_root_dir ./logs_store/deepspeed/ \
    --train_datadir $TRAIN_DIR \
    --valid_datadir $VALID_DIR \
    --log_dir logs \
    --seed 42 \
    --lr $LR \
    --weight_decay 0.1 \
    --gradient_clip_val 1.0 \
    --max_epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --train_batch_size $TRAIN_BS \
    --valid_batch_size $VALID_BS \
    --accumulate_grad_batches $GRAD_STEPS \
    --log_every_n_steps 100 \
    --save_step_frequency 1000 \
    --val_check_interval 1 \
    --debug_cuda_mem \
    --use_deepspeed \
    --precision 16  \
    $CL_Config
