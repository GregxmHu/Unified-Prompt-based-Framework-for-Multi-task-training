set -ex
export CUDA_VISIBLE_DEVICES=0,3
LR=2e-5

MAX_STEPS=1000
EPOCH=6

LOG_STEP=10
EVAL_EVERY=100

BATCH_SIZE=8


pretrained_ckpt=""
# pretrained_ckpt="/data/private/yushi/pretrained_models/t5-large"
dir_path=""
python -m torch.distributed.launch \
         --nproc_per_node=2 \
         --master_port=21227  \
        train.py \
        -train $dir_path/dataset/mnli/train.jsonl  \
        -max_input 80000000  \
        -save $dir_path/checkpoints/mnli/  \
        -dev $dir_path/dataset/mnli/val_mismatch.jsonl   \
        -vocab $pretrained_ckpt          \
        -pretrain $pretrained_ckpt   \
        -res results.jsonl  \
        -epoch $EPOCH  \
        -n_warmup_steps 0  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -gradient_accumulation_steps 8 \
        -dev_eval_batch_size 128  \
        -eval_every $EVAL_EVERY  \
        -optimizer adamw  \
        -logging_step $LOG_STEP  \
        --max_steps=$MAX_STEPS \
        --template="mnli hypothesis: <h> premise: <p> entailment: "
        --prefix=''   \
        --infix=''    \
        --suffix=''   \
       


