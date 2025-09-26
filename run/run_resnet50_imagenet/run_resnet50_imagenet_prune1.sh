#!/bin/bash
# Default values
arch=${ARCH:-ResNet_50}
result_dir=${RESULT_DIR:-/kaggle/working/results/run_resnet50_imagenet_prune1}
teacher_ckpt_path=${TEACHER_CKPT_PATH:-/kaggle/working/KDFS/teacher_dir/teacher_model_best.pth}
num_gpus=${NUM_GPUS:-1}
num_workers=${NUM_WORKERS:-4}
pin_memory=${PIN_MEMORY:-true}
seed=${SEED:-3407}
lr=${LR:-0.004}
warmup_steps=${WARMUP_STEPS:-10}
warmup_start_lr=${WARMUP_START_LR:-1e-05}
lr_decay_T_max=${LR_DECAY_T_MAX:-250}
lr_decay_eta_min=${LR_DECAY_ETA_MIN:-4e-05}
weight_decay=${WEIGHT_DECAY:-0.0005}
train_batch_size=${TRAIN_BATCH_SIZE:-32}
eval_batch_size=${EVAL_BATCH_SIZE:-32}
target_temperature=${TARGET_TEMPERATURE:-3}
gumbel_start_temperature=${GUMBEL_START_TEMPERATURE:-1}
gumbel_end_temperature=${GUMBEL_END_TEMPERATURE:-0.1}
coef_kdloss=${COEF_KDLOSS:-0.5}
coef_rcloss=${COEF_RCLOSS:-1.0}
coef_maskloss=${COEF_MASKLOSS:-1.0}
compress_rate=${COMPRESS_RATE:-0.3}
finetune_num_epochs=${FINETUNE_NUM_EPOCHS:-15}
finetune_lr=${FINETUNE_LR:-4e-06}
finetune_warmup_steps=${FINETUNE_WARMUP_STEPS:-5}
finetune_warmup_start_lr=${FINETUNE_WARMUP_START_LR:-4e-08}
finetune_lr_decay_T_max=${FINETUNE_LR_DECAY_T_MAX:-20}
finetune_lr_decay_eta_min=${FINETUNE_LR_DECAY_ETA_MIN:-4e-08}
finetune_weight_decay=${FINETUNE_WEIGHT_DECAY:-2e-05}
finetune_train_batch_size=${FINETUNE_TRAIN_BATCH_SIZE:-128}
finetune_eval_batch_size=${FINETUNE_EVAL_BATCH_SIZE:-128}
dataset_mode=${DATASET_MODE:-hardfake}
dataset_dir=${DATASET_DIR:-/kaggle/input/hardfakevsrealfaces}
master_port=${MASTER_PORT:-6681}
num_epochs=${NUM_EPOCHS:-3}
resume=${RESUME:-}

# Environment variables for CUDA and memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Set CUDA_VISIBLE_DEVICES based on num_gpus
if [ "$num_gpus" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$num_gpus" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
else
    echo "Error: Invalid number of GPUs. Use 1 or 2."
    exit 1
fi

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Check phase argument
PHASE=${1:-train}
shift
if [[ "$PHASE" != "train" && "$PHASE" != "finetune" ]]; then
    echo "Error: Invalid phase. Use 'train' or 'finetune'."
    exit 1
fi

# Create result directory
mkdir -p "$result_dir"

# Set nproc_per_node
nproc_per_node=$num_gpus

# Define pin_memory flag
pin_memory_flag=""
if [ "$pin_memory" = "true" ]; then
    pin_memory_flag="--pin_memory"
fi

# Parse additional arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --num_gpus) num_gpus="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --num_epochs) num_epochs="$2"; shift 2 ;;
        --finetune_num_epochs) finetune_num_epochs="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --gumbel_start_temperature) gumbel_start_temperature="$2"; shift 2 ;;
        --gumbel_end_temperature) gumbel_end_temperature="$2"; shift 2 ;;
        --warmup_start_lr) warmup_start_lr="$2"; shift 2 ;;
        --coef_kdloss) coef_kdloss="$2"; shift 2 ;;
        --coef_rcloss) coef_rcloss="$2"; shift 2 ;;
        --coef_maskloss) coef_maskloss="$2"; shift 2 ;;
        --compress_rate) compress_rate="$2"; shift 2 ;;
        --warmup_steps) warmup_steps="$2"; shift 2 ;;
        --dataset_mode) dataset_mode="$2"; shift 2 ;;
        --dataset_dir) dataset_dir="$2"; shift 2 ;;
        --teacher_ckpt_path) teacher_ckpt_path="$2"; shift 2 ;;
        --resume) resume="$2"; shift 2 ;;
        --train_batch_size) train_batch_size="$2"; shift 2 ;;
        --eval_batch_size) eval_batch_size="$2"; shift 2 ;;
        --lr_decay_T_max) lr_decay_T_max="$2"; shift 2 ;;
        --finetune_student_ckpt_path) finetune_student_ckpt_path="$2"; shift 2 ;;
        --finetune_train_batch_size) finetune_train_batch_size="$2"; shift 2 ;;
        --finetune_eval_batch_size) finetune_eval_batch_size="$2"; shift 2 ;;
        --ddp) ddp_flag="--ddp"; shift ;;
        *) echo "Ignoring unrecognized argument: $1"; shift ;;
    esac
done

# Re-validate CUDA_VISIBLE_DEVICES
if [ "$num_gpus" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$num_gpus" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
else
    echo "Error: Invalid number of GPUs. Use 1 or 2."
    exit 1
fi

# Check if teacher checkpoint exists
if [ ! -f "$teacher_ckpt_path" ]; then
    echo "Error: Teacher checkpoint not found at $teacher_ckpt_path"
    exit 1
fi

# Check if resume checkpoint exists (if provided)
if [ -n "$resume" ] && [ ! -f "$resume" ]; then
    echo "Error: Resume checkpoint not found at $resume"
    exit 1
fi

if [ "$PHASE" = "train" ]; then
    echo "Running training phase with train_batch_size=$train_batch_size, eval_batch_size=$eval_batch_size"
    torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port /kaggle/working/KDFS/main.py \
        --phase train \
        --arch "$arch" \
        --device cuda \
        --result_dir "$result_dir" \
        --teacher_ckpt_path "$teacher_ckpt_path" \
        --num_workers "$num_workers" \
        $pin_memory_flag \
        --seed "$seed" \
        --num_epochs "$num_epochs" \
        --lr "$lr" \
        --warmup_steps "$warmup_steps" \
        --warmup_start_lr "$warmup_start_lr" \
        --lr_decay_T_max "$lr_decay_T_max" \
        --lr_decay_eta_min "$lr_decay_eta_min" \
        --weight_decay "$weight_decay" \
        --train_batch_size "$train_batch_size" \
        --eval_batch_size "$eval_batch_size" \
        --target_temperature "$target_temperature" \
        --gumbel_start_temperature "$gumbel_start_temperature" \
        --gumbel_end_temperature "$gumbel_end_temperature" \
        --coef_kdloss "$coef_kdloss" \
        --coef_rcloss "$coef_rcloss" \
        --coef_maskloss "$coef_maskloss" \
        --compress_rate "$compress_rate" \
        --dataset_mode "$dataset_mode" \
        --dataset_dir "$dataset_dir" \
        $( [ -n "$resume" ] && echo "--resume $resume" ) \
        $ddp_flag

elif [ "$PHASE" = "finetune" ]; then
    # اعتبارسنجی دوباره CUDA_VISIBLE_DEVICES برای فاین‌تیونینگ
    if [ "$num_gpus" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [ "$num_gpus" -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    else
        echo "خطا: تعداد GPU نامعتبر است. از 1 یا 2 استفاده کنید."
        exit 1
    fi

    student_ckpt_path="${finetune_student_ckpt_path:-$result_dir/student_model/${arch}_sparse_best.pt}"
    if [ ! -f "$student_ckpt_path" ]; then
        echo "خطا: فایل چک‌پوینت دانش‌آموز در $student_ckpt_path یافت نشد"
        exit 1
    fi

    echo "اجرای فاز فاین‌تیونینگ با finetune_train_batch_size=$finetune_train_batch_size, finetune_eval_batch_size=$finetune_eval_batch_size"
    echo "چک‌پوینت دانش‌آموز: $student_ckpt_path"
    torchrun --nproc_per_node="$nproc_per_node" --master_port="$master_port" /kaggle/working/KDFS/main.py \
        --phase finetune \
        --arch "$arch" \
        --device cuda \
        --result_dir "$result_dir" \
        --teacher_ckpt_path "$teacher_ckpt_path" \
        --finetune_student_ckpt_path "$student_ckpt_path" \
        --num_workers "$num_workers" \
        $pin_memory_flag \
        --seed "$seed" \
        --finetune_num_epochs "$finetune_num_epochs" \
        --finetune_lr "$finetune_lr" \
        --finetune_warmup_steps "$finetune_warmup_steps" \
        --finetune_warmup_start_lr "$finetune_warmup_start_lr" \
        --finetune_lr_decay_T_max "$finetune_lr_decay_T_max" \
        --finetune_lr_decay_eta_min "$finetune_lr_decay_eta_min" \
        --finetune_weight_decay "$finetune_weight_decay" \
        --train_batch_size "$finetune_train_batch_size" \
        --eval_batch_size "$finetune_eval_batch_size" \
        --sparsed_student_ckpt_path "$result_dir/student_model/finetune_${arch}_sparse_best.pt" \
        --dataset_mode "$dataset_mode" \
        --dataset_dir "$dataset_dir" \
        $( [ -n "$resume" ] && echo "--resume $resume" ) \
        $ddp_flag
fi
