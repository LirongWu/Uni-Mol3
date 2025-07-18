export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export NCCL_NVLS_ENABLE=0
export MASTER_PORT=$1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)

cd /wulirong/Uni-Mol3
save_dir="./save_dir/finetune"

wd=1e-4
max_steps=1500000
epoch=5000

batch_size=64
update_freq=1
seed=1

task_type=$2
if [ "$task_type" = "yield" ]; then
    loss_func=unimol3_regression
else
    loss_func=unimol3_T5
fi

lr=5e-4
data_path='./data/pistachio/full_data/'
arch="T5_base"
warmup_steps=0
interval=100000
weight_path='./model/Reaction_Pretrained/checkpoint.pt'
valid_set=test_forward


timestamp=$(date '+%Y%m%d_%H%M%S')
branch=$(git branch --show-current)
commit_id=$(git rev-parse --short HEAD)
save_dir="${save_dir}/${timestamp}"
log_filename="training_log.txt"
mkdir -p $save_dir


torchrun --standalone --nnodes=1 --nproc_per_node=$n_gpu --master_port=$MASTER_PORT \
    $(which unicore-train) $data_path \
    --user-dir "unimol3" --train-subset train --valid-subset $valid_set \
    --num-workers 8 --ddp-backend=no_c10d \
    --task unimol3_T5 --loss $loss_func --arch $arch \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
    --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
    --update-freq $update_freq --seed $seed \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
    --max-update $max_steps --log-interval 10 --log-format simple \
    --validate-interval-updates $max_steps --save-interval-updates $interval --keep-interval-updates 10 \
    --validate-interval 20 --save-interval 1 --keep-last-epochs 0 \
    --batch-size $batch_size \
    --max-epoch $epoch \
    --task-type $task_type \
    --weight-dir-path $weight_path \
    --tmp-save-dir $save_dir --save-dir $save_dir 2>&1 | tee ${save_dir}/${log_filename}