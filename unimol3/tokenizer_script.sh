export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MASTER_PORT=$1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)

cd /wulirong/Uni-Mol3
data_path=$2
save_dir="./save_dir/tokenizer"
weight_path="./model/UniMol2_84M/checkpoint.pt"

lr=1e-4
wd=1e-4
warmup_steps=50000
max_steps=10000000

batch_size=32
update_freq=1
masked_token_loss=1
masked_coord_loss=1
masked_dist_loss=1
mask_prob=1.0
noise_type="uniform"
noise=0.2
seed=1

mask_token_prob=0.0
drop_feat_prob=0.5

ema_decay=0.999
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${pair_dropout}" ] && pair_dropout=0.25


timestamp=$(date '+%Y%m%d_%H%M%S')
branch=$(git branch --show-current)
commit_id=$(git rev-parse --short HEAD)
save_dir="${save_dir}/${timestamp}"
log_filename="training_log.txt"
mkdir -p $save_dir


torchrun --standalone --nnodes=1 --nproc_per_node=$n_gpu --master_port=$MASTER_PORT \
    $(which unicore-train) $data_path \
    --finetune-from-model $weight_path --tokenization True \
    --user-dir "unimol3" --train-subset train --valid-subset valid \
    --num-workers 8 --ddp-backend=no_c10d \
    --task unimol3_tokenizer --loss unimol3_tokenizer --arch unimol3_84M \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
    --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
    --update-freq $update_freq --seed $seed \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
    --max-update $max_steps --log-interval 10 --log-format simple \
    --validate-interval-updates 10000 --save-interval-updates 500000 --keep-interval-updates 10  \
    --validate-interval 1 --save-interval 1 --keep-last-epochs 0 \
    --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
    --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
    --mask-token-prob $mask_token_prob \
    --drop-feat-prob $drop_feat_prob \
    --droppath-prob $droppath_prob  \
    --pair-dropout $pair_dropout \
    --ema-decay $ema_decay --validate-with-ema \
    --max-epoch 80 \
    --tmp-save-dir $save_dir --save-dir $save_dir 2>&1 | tee ${save_dir}/${log_filename}