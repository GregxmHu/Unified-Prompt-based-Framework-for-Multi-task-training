set -ex
export CUDA_VISIBLE_DEVICES=3,4
LR=1e-4

MAX_STEPS=80000
EPOCH=6

LOG_STEP=100
EVAL_EVERY=500

BATCH_SIZE=8


pretrained_ckpt="/home/huxiaomeng/t5v11large/"
# pretrained_ckpt="/data/private/yushi/pretrained_models/t5-large"

python -m torch.distributed.launch \
         --nproc_per_node=2 \
         --master_port=21227  \
        train.py \
        -train /data/private/huxiaomeng/promptir/dataset/mnli/train.jsonl  \
        -max_input 80000000  \
        -save /data/private/huxiaomeng/promptir/checkpoints/mnli/  \
        -dev /data/private/huxiaomeng/promptir/dataset/mnli/val_mismatch.jsonl   \
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
        
       


