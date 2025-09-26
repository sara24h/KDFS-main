arch=resnet_56
result_dir=result/run_resnet56_hardfakevsrealfaces
dataset_dir=/kaggle/input/hardfakevsrealfaces
dataset_type=hardfakevsrealfaces
teacher_ckpt_path=teacher_resnet56_finetuned.pth
device=0

CUDA_VISIBLE_DEVICES=$device python main.py \
--phase train \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--num_workers 4 \
--pin_memory \
--device cuda \
--arch $arch \
--seed 3407 \
--result_dir $result_dir \
--teacher_ckpt_path $teacher_ckpt_path \
--num_epochs 50 \
--lr 1e-3 \
--warmup_steps 5 \
--warmup_start_lr 1e-5 \
--lr_decay_T_max 50 \
--lr_decay_eta_min 1e-5 \
--weight_decay 2e-5 \
--train_batch_size 32 \
--eval_batch_size 32 \
--target_temperature 3 \
--gumbel_start_temperature 1 \
--gumbel_end_temperature 0.1 \
--coef_kdloss 0.05 \
--coef_rcloss 1000 \
--coef_maskloss 10000 \
--compress_rate 0.5 \
&& \
CUDA_VISIBLE_DEVICES=$device python main.py \
--phase finetune \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--num_workers 4 \
--pin_memory \
--device cuda \
--arch $arch \
--seed 3407 \
--result_dir $result_dir \
--finetune_student_ckpt_path $result_dir"/student_model/"$arch"_sparse_last.pt" \
--finetune_num_epochs 10 \
--finetune_lr 1e-5 \
--finetune_warmup_steps 2 \
--finetune_warmup_start_lr 1e-7 \
--finetune_lr_decay_T_max 10 \
--finetune_lr_decay_eta_min 1e-7 \
--finetune_weight_decay 2e-5 \
--finetune_train_batch_size 32 \
--finetune_eval_batch_size 32 \
--sparsed_student_ckpt_path $result_dir"/student_model/finetune_"$arch"_sparse_best.pt"
