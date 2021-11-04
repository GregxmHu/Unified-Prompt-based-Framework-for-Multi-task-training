set -ex
export CUDA_VISIBLE_DEVICES=2,3,7
LR=2e-5

MAX_STEPS=80000
EPOCH=6

LOG_STEP=50
EVAL_EVERY=50

BATCH_SIZE=4


pretrained_ckpt="/home/huxiaomeng/t5v11large/"

python -m torch.distributed.launch \
         --nproc_per_node=3 \
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
        --log_dir=/data/private/huxiaomeng/promptir/logs/testt5v11/q$Q-n-$NEG/ \
        --max_steps=$MAX_STEPS \
        
       


