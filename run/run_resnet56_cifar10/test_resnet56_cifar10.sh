
arch=resnet_56
dataset_dir="/kaggle/input/hardfakevsrealfaces"
dataset_type=hardfakevsrealfaces
ckpt_path="/kaggle/working/KDFS/result/run_resnet56_cifar10_prune1/student_model/finetune_resnet_56_sparse_best.pt"
device=0
CUDA_VISIBLE_DEVICES=$device python main.py \
--phase test \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--num_workers 8 \
--pin_memory \
--device cuda \
--arch $arch \
--test_batch_size 256 \
--sparsed_student_ckpt_path $ckpt_path \
