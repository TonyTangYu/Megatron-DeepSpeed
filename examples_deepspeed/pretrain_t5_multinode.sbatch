#!/bin/bash
#SBATCH --job-name=t5_layer
#SBATCH --output=t5/t5_4_3090_layer_PP2.out
#SBATCH -p ai
#SBATCH --nodes=4
#SBATCH -w gn[11-12,33-34]
#SBATCH --ntasks-per-node=4
#SBATCH --account=zlz
#SBATCH --mem=MaxMemPerNode


export CUDA_DEVICE_MAX_CONNECTIONS=1

export GPUS_PER_NODE=4
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
echo $MASTER_ADDR
echo $MASTER_PORT
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile


echo "WORLD_SIZE:" $WORLD_SIZE "NODE_RANK:" $NODE_RANK "DEVICES:" $CUDA_VISIBLE_DEVICES
echo "MASTER_ADDR:" $MASTER_ADDR "MASTER_PORT:" $MASTER_PORT

TP=1
PP=4  
GLOBAL_BATCH=32
MICRO_BATCH=4
ZERO_STAGE=0
DS_CONFIG=deepspeed.json
VOCAB_FILE=/gf3/home/zlz/ty/bert_pretrain/bert_vocab.txt
DATA_PATH=/gf3/home/zlz/ty/bert_pretrain/my-bert_text_sentence/my-bert_text_sentence

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "fp16": {
    "enabled": false,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

T5_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --pipeline-model-parallel-split-rank 2 \
    --decoder-num-layers 6  \
    --num-layers 24 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --vocab-extra-ids 100
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

# torchrun $DISTRIBUTED_ARGS pretrain_t5.py \
deepspeed --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --hostfile "./hostfile" \
    --no_ssh_check \
    --launcher SLURM \
    ../pretrain_t5.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --no-masked-softmax-fusion \
    --pipeline-partition-method uniform \
