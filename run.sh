set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
LR=2e-5

MAX_STEPS=20000
EPOCH=10000000

LOG_STEP=100
EVAL_EVERY=100

BATCH_SIZE=4
#checkpoints="/data/private/huxiaomeng/checkpoints/mnli/_step-12000.bin"
pretrained_ckpt="/data/private/huxiaomeng/pretrained_models/t5-v11-large"
# pretrained_ckpt="/data/private/yushi/pretrained_models/t5-large"
dir_path="/data/private/huxiaomeng/promptir"
python -m torch.distributed.launch \
         --nproc_per_node=4 \
         --master_port=21227  \
        train.py \
        -train $dir_path/dataset/mnli/train.jsonl  \
        -max_input 80000000  \
        -save $dir_path/checkpoints/mnli/  \
        -dev $dir_path/dataset/mnli/val_mismatch.jsonl   \
        -vocab $pretrained_ckpt          \
        -pretrain $pretrained_ckpt   \
        -res $dir_path/results/mnli_results.jsonl  \
        -epoch $EPOCH  \
        -n_warmup_steps 0  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -gradient_accumulation_steps 1 \
        -dev_eval_batch_size 128  \
        -eval_every $EVAL_EVERY  \
        -optimizer adamw  \
        -logging_step $LOG_STEP  \
        --max_steps=$MAX_STEPS \
        -template "mnli hypothesis: <h> premise: <p> entailment: "	\
        --prefix="[1,2,3]"   \
        --infix="[4,5,6]"    \
        --suffix="[7,8,9]"   \
        --original_t5   \
	    #-checkpoint $checkpoints \
       


