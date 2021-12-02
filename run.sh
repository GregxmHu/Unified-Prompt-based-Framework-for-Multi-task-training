set -ex
export CUDA_VISIBLE_DEVICES=0,3
LR=2e-5

MAX_STEPS=1000
EPOCH=6

LOG_STEP=10
EVAL_EVERY=100

BATCH_SIZE=8


pretrained_ckpt="t5-large"
# pretrained_ckpt="/data/private/yushi/pretrained_models/t5-large"

python -m torch.distributed.launch \
         --nproc_per_node=2 \
         --master_port=21227  \
        train.py \
        -train /data/private/huxiaomeng/promptir/dataset/mnli/train.jsonl  \
        -max_input 80000000  \
        -save /data/private/huxiaomeng/promptir/checkpoints/mnli/  \
        -dev /data/private/huxiaomeng/promptir/dataset/mnli/val_mismatch.jsonl   \
        -test /data/private/
        -vocab $pretrained_ckpt          \
        -pretrain $pretrained_ckpt   \
        -res results.jsonl  \
        -epoch $EPOCH  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -eval_every $EVAL_EVERY  \
        -optimizer adamw  \
        -dev_eval_batch_size 128  \
        -n_warmup_steps 0  \
        -logging_step $LOG_STEP  \
        --max_steps=$MAX_STEPS \
        --original_t5
        
       


