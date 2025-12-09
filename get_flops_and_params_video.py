import torch
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv
# --- تغییر: ایمپورت مدل prune شده برای UADFV ---
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_uadfv, ResNet_50_pruned_hardfakevsreal
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_deepfake
from model.pruned_model.MobileNetV2_pruned import MobileNetV2_pruned
from model.student.GoogleNet_sparse import GoogLeNet_sparse_deepfake
from model.pruned_model.GoogleNet_pruned import GoogLeNet_pruned_deepfake
from thop import profile

# Base FLOPs and parameters for each dataset
Flops_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 7700.0,
        "rvf10k": 5390.0,
        "140k": 5390.0,
        "190k": 5390.0,
        "200k": 5390.0,
        "330k": 5390.0,
        "125k": 2100.0,
        "uadfv": 172690,
    },
    "MobileNetV2": {
        "hardfakevsrealfaces": 570.0,
        "rvf10k": 416.68,
        "140k": 416.68,
        "200k": 416.68,
        "330k": 416.68,
        "190k": 416.68,
        "125k": 153.0,
    },
    "googlenet": {
        "hardfakevsrealfaces": 570.0,
        "rvf10k": 1980,
        "140k": 1980,
        "200k": 1980,
        "330k": 1980,
        "190k": 1980,
        "125k": 153.0,
    }
}
Params_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 14.97,
        "rvf10k": 23.51,
        "140k": 23.51,
        "190k": 23.51,
        "200k": 23.51,
        "330k": 23.51,
        "125k": 23.51,
        "uadfv": 23.51, 
    },
    "MobileNetV2": {
        "hardfakevsrealfaces": 2.23,
        "rvf10k": 2.23,
        "140k": 2.23,
        "200k": 2.23,
        "330k": 2.23,
        "190k": 2.23,
        "125k": 2.23,
    },
    "googlenet": {
        "hardfakevsrealfaces": 2.23,
        "rvf10k": 5.6,
        "140k": 5.6,
        "200k": 5.6,
        "330k": 5.6,
        "190k": 5.6,
        "125k": 2.23,
    }
}
image_sizes = {
    "hardfakevsreal": 300,
    "rvf10k": 256,
    "140k": 256,
    "190k": 256,
    "200k": 256,
    "330k": 256,
    "125k": 160,
    "uadfv": 256
}

num_frames_per_clip = {
    "uadfv": 32,  

}

def get_flops_and_params(dataset_mode, sparsed_student_ckpt_path):
    # Map dataset_mode to dataset_type
    dataset_type = {
        "hardfake": "hardfakevsreal",
        "rvf10k": "rvf10k",
        "140k": "140k",
        "190k": "190k",
        "200k": "200k",
        "330k": "330k",
        "125k": "125k",
        "uadfv": "uadfv",
    }[dataset_mode]

    num_frames = num_frames_per_clip.get(dataset_type, 1)

    # Load checkpoint
    ckpt_student = torch.load(sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt_student["student"]

    # Determine model type based on state_dict keys
    if any(key.startswith('features.') for key in state_dict):
        model_type = "MobileNetV2"
        student = MobileNetV2_sparse_deepfake()
        
    elif any(key.startswith('inception3a.') for key in state_dict): 
        model_type = "googlenet"
        student = GoogLeNet_sparse_deepfake()
 
    else:
        model_type = "ResNet_50"
        student = ResNet_50_sparse_hardfakevsreal()
        
    student.load_state_dict(state_dict)

    # Adjust dataset_type for MobileNetV2 if necessary
    if model_type in ["MobileNetV2", "googlenet"] and dataset_type == "hardfakevsreal":
        dataset_type = "hardfakevsrealfaces"

    # Extract masks
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]

    if model_type == "ResNet_50":
        if dataset_type == "uadfv":
            pruned_model = ResNet_50_pruned_uadfv(masks=masks)
        else:
            pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    elif model_type == "MobileNetV2":
        pruned_model = MobileNetV2_pruned(masks=masks)
    elif model_type == "googlenet":
        pruned_model = GoogLeNet_pruned_deepfake(masks=masks)
    
    # Set input size based on dataset (still a single frame for the model)
    input = torch.rand([1, 3, image_sizes[dataset_type], image_sizes[dataset_type]])
    
    # Calculate FLOPs for a single frame
    Flops_per_frame, Params = profile(pruned_model, inputs=(input,), verbose=False)

    # --- تغییر: محاسبه FLOPs کل برای کلیپ ویدیویی ---
    Flops = Flops_per_frame * num_frames

    # Use dataset-specific baseline values
    Flops_baseline = Flops_baselines[model_type][dataset_type]
    Params_baseline = Params_baselines[model_type][dataset_type]

    Flops_reduction = (
        (Flops_baseline - Flops / (10**6)) / Flops_baseline * 100.0
    )
    Params_reduction = (
        (Params_baseline - Params / (10**6)) / Params_baseline * 100.0
    )
    return (
        Flops_baseline,
        Flops / (10**6),
        Flops_reduction,
        Params_baseline,
        Params / (10**6),
        Params_reduction,
    )

def main():
    sparsed_student_ckpt_path = None  # Set your checkpoint path here, e.g., "path/to/your/ckpt.pth"

    if sparsed_student_ckpt_path is None:
        raise ValueError("Please set the sparsed_student_ckpt_path in the code.")

    # Run for all datasets
    for dataset_mode in ["hardfake", "rvf10k", "140k", "190k", "200k", "330k", "125k","uadfv"]:
        print(f"\nEvaluating for dataset: {dataset_mode}")
        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(dataset_mode, sparsed_student_ckpt_path)
        print(
            "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
            % (Params_baseline, Params, Params_reduction)
        )
        print(
            "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
            % (Flops_baseline, Flops, Flops_reduction)
        )

if __name__ == "__main__":
    main()
