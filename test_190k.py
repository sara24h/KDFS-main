import os
import random
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter


def set_seed(seed: int):
    """Sets the seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to {seed} for reproducibility.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Sparse ResNet50 on ANY dataset")
    
    # ✅ مسیر چک‌پوینت
    parser.add_argument('--sparsed_student_ckpt_path', type=str, required=True,
                        help='Path to the pruned student checkpoint')
    
    # ✅ دیتاست
    parser.add_argument('--dataset_mode', type=str, default="rvf10k",
                        choices=['rvf10k', '190k', '140k', '200k', 'hardfake', '330k'],
                        help='Which dataset to use')
    
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Custom dataset path (optional, auto-detected)')
    
    parser.add_argument('--new_dataset_dir', type=str, default=None,
                        help='External test dataset (e.g. Celeb-DF, FF++)')
    
    # ✅ هایپرپارامترهای Fine-tuning (اصلاح شده)
    parser.add_argument('--f_lr', type=float, default=1e-4,
                        help='Fine-tuning learning rate')
    
    parser.add_argument('--f_epochs', type=int, default=12,
                        help='Number of fine-tuning epochs')
    
    parser.add_argument('--f_weight_decay', type=float, default=1e-5,
                        help='Weight decay for fine-tuning')
    
    # ✅ بچ سایز
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    
    # ✅ مسیر خروجی
    parser.add_argument('--result_dir', type=str, default="/kaggle/working/results")
    
    return parser.parse_args()


# مسیرهای خودکار برای دیتاست‌های کگل
AUTO_PATHS = {
    "rvf10k": "/kaggle/input/rvf10k",
    "190k": "/kaggle/input/deepfake-and-real-images/Dataset",
    "140k": "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake",
    "200k": "/kaggle/input/200k-real-and-fake-faces",
    "hardfake": "/kaggle/input/hardfakevsrealfaces",
    "330k": "/kaggle/input/deepfake-dataset",
}


class Config:
    def __init__(self):
        args = parse_args()
        
        self.seed = 3407
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_mode = args.dataset_mode
        
        # ✅ خودکار پیدا کردن مسیر دیتاست
        self.dataset_dir = args.dataset_dir or AUTO_PATHS.get(self.dataset_mode, None)
        if self.dataset_dir is None:
            raise ValueError(f"Dataset '{self.dataset_mode}' not found! Use --dataset_dir")
        
        self.new_dataset_dir = args.new_dataset_dir
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.result_dir = args.result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
        # ✅ بچ سایز
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = 4
        self.pin_memory = True
        
        # ✅ هایپرپارامترها
        self.f_epochs = args.f_epochs
        self.f_lr = args.f_lr
        self.weight_decay = args.f_weight_decay
        
        # چاپ تنظیمات
        print("=" * 60)
        print(f"🎯 Dataset Mode → {self.dataset_mode}")
        print(f"📁 Dataset Path → {self.dataset_dir}")
        print(f"💾 Checkpoint → {self.sparsed_student_ckpt_path}")
        print(f"📊 Output Dir → {self.result_dir}")
        print(f"🔧 Fine-tuning: epochs={self.f_epochs}, lr={self.f_lr}, wd={self.weight_decay}")
        print("=" * 60)


class Test:
    def __init__(self, config):
        self.config = config
        set_seed(self.config.seed)
        
        self.dataset_dir = self.config.dataset_dir
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory
        self.arch = 'resnet50'
        self.device = self.config.device
        self.train_batch_size = self.config.train_batch_size
        self.test_batch_size = self.config.test_batch_size
        self.sparsed_student_ckpt_path = self.config.sparsed_student_ckpt_path
        self.dataset_mode = self.config.dataset_mode
        self.result_dir = self.config.result_dir
        self.new_dataset_dir = self.config.new_dataset_dir
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None
        self.student = None

    def dataload(self):
        print("==> Loading datasets...")
        
        image_size = (256, 256)
        mean_190k = [0.4668, 0.3816, 0.3414]
        std_190k = [0.2410, 0.2161, 0.2081]
        
        transform_train_190k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_190k, std=std_190k),
        ])
        
        transform_val_test_190k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_190k, std=std_190k),
        ])
        
        params = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'ddp': False
        }
        
        # ✅ تنظیم صحیح پارامترها بر اساس دیتاست
        if self.dataset_mode == 'hardfake':
            params['hardfake_csv_file'] = os.path.join(self.dataset_dir, 'data.csv')
            params['hardfake_root_dir'] = self.dataset_dir
            
        elif self.dataset_mode == 'rvf10k':
            params['rvf10k_train_csv'] = os.path.join(self.dataset_dir, 'train.csv')
            params['rvf10k_valid_csv'] = os.path.join(self.dataset_dir, 'valid.csv')
            params['rvf10k_root_dir'] = self.dataset_dir
            
        elif self.dataset_mode == '140k':
            params['realfake140k_train_csv'] = os.path.join(self.dataset_dir, 'train.csv')
            params['realfake140k_valid_csv'] = os.path.join(self.dataset_dir, 'valid.csv')
            params['realfake140k_test_csv'] = os.path.join(self.dataset_dir, 'test.csv')
            params['realfake140k_root_dir'] = self.dataset_dir
            
        elif self.dataset_mode == '200k':
            image_root_dir = os.path.join(self.dataset_dir, 'my_real_vs_ai_dataset', 'my_real_vs_ai_dataset')
            params['realfake200k_root_dir'] = image_root_dir
            params['realfake200k_train_csv'] = os.path.join(self.dataset_dir, 'train_labels.csv')
            params['realfake200k_val_csv'] = os.path.join(self.dataset_dir, 'val_labels.csv')
            params['realfake200k_test_csv'] = os.path.join(self.dataset_dir, 'test_labels.csv')
            
        elif self.dataset_mode == '190k':
            params['realfake190k_root_dir'] = self.dataset_dir
            
        elif self.dataset_mode == '330k':
            params['realfake330k_root_dir'] = self.dataset_dir
        
        print(f"📦 Creating Dataset_selector with params: {list(params.keys())}")
        dataset_manager = Dataset_selector(**params)
        
        print("✅ Overriding transforms to use 190k normalization")
        dataset_manager.loader_train.dataset.transform = transform_train_190k
        dataset_manager.loader_val.dataset.transform = transform_val_test_190k
        dataset_manager.loader_test.dataset.transform = transform_val_test_190k
        
        self.train_loader = dataset_manager.loader_train
        self.val_loader = dataset_manager.loader_val
        self.test_loader = dataset_manager.loader_test
        
        print(f"✅ Loaders configured: Train={len(self.train_loader)} batches, "
              f"Val={len(self.val_loader)} batches, Test={len(self.test_loader)} batches")

    def build_model(self):
        print("==> Building student model...")
        self.student = ResNet_50_sparse_hardfakevsreal()
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.sparsed_student_ckpt_path}")
        
        print(f"Loading weights from: {self.sparsed_student_ckpt_path}")
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt_student.get("student", ckpt_student)
        
        self.student.load_state_dict(state_dict, strict=False)
        
        # Add dropout
        self.student.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            self.student.fc
        )
        
        self.student.to(self.device)
        print(f"✅ Model loaded on {self.device}")

    def compute_metrics(self, loader, description="Test", print_metrics=True, save_confusion_matrix=True):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        all_preds = []
        all_targets = []
        sample_info = []
        
        self.student.eval()
        self.student.ticket = True
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=description, ncols=100)):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                
                logits, _ = self.student(images)
                logits = logits.squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                batch_size = images.size(0)
                for i in range(batch_size):
                    try:
                        img_path = loader.dataset.samples[batch_idx * loader.batch_size + i][0]
                    except (AttributeError, IndexError):
                        img_path = f"Sample_{batch_idx * loader.batch_size + i}"
                    
                    sample_info.append({
                        'id': img_path,
                        'true_label': targets[i].item(),
                        'pred_label': preds[i].item()
                    })
                
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                meter_top1.update(prec1, images.size(0))
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        accuracy = meter_top1.avg
        precision = precision_score(all_targets, all_preds, average='binary')
        recall = recall_score(all_targets, all_preds, average='binary')
        
        precision_per_class = precision_score(all_targets, all_preds, average=None, labels=[0, 1])
        recall_per_class = recall_score(all_targets, all_preds, average=None, labels=[0, 1])
        
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
        specificity_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if print_metrics:
            print(f"\n[{description}] Overall Metrics:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  Specificity: {specificity_real:.4f}")
            
            print(f"\n[{description}] Per-Class Metrics:")
            print(f"  Class Real (0):")
            print(f"    Precision: {precision_per_class[0]:.4f}")
            print(f"    Recall: {recall_per_class[0]:.4f}")
            print(f"    Specificity: {specificity_real:.4f}")
            print(f"  Class Fake (1):")
            print(f"    Precision: {precision_per_class[1]:.4f}")
            print(f"    Recall: {recall_per_class[1]:.4f}")
            print(f"    Specificity: {specificity_fake:.4f}")
        
        cm = confusion_matrix(all_targets, all_preds)
        classes = ['Real', 'Fake']
        
        if save_confusion_matrix:
            print(f"\n[{description}] Confusion Matrix:")
            print(f"{'':>10} {'Predicted Real':>15} {'Predicted Fake':>15}")
            print(f"{'Actual Real':>10} {cm[0,0]:>15} {cm[0,1]:>15}")
            print(f"{'Actual Fake':>10} {cm[1,0]:>15} {cm[1,1]:>15}")
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix - {description}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            sanitized_description = description.lower().replace(" ", "_").replace("/", "_")
            plot_path = os.path.join(self.result_dir, f'confusion_matrix_{sanitized_description}.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
            print(f"✅ Confusion matrix saved: {plot_path}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity_real,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'specificity_per_class': [specificity_real, specificity_fake],
            'confusion_matrix': cm,
            'sample_info': sample_info
        }

    def display_samples(self, sample_info, description="Test", num_samples=30):
        print(f"\n[{description}] First {num_samples} samples:")
        print(f"{'Sample ID':<50} {'True Label':<12} {'Predicted Label':<12}")
        print("-" * 80)
        for i, sample in enumerate(sample_info[:num_samples]):
            true_label = 'Real' if sample['true_label'] == 0 else 'Fake'
            pred_label = 'Real' if sample['pred_label'] == 0 else 'Fake'
            print(f"{sample['id']:<50} {true_label:<12} {pred_label:<12}")

    def finetune(self):
        print("==> Fine-tuning (FEATURE EXTRACTOR strategy on fc + layer4)...")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        for name, param in self.student.named_parameters():
            if 'fc' in name or 'layer4' in name:
                param.requires_grad = True
                print(f"  ✅ Unfreezing: {name}")
            else:
                param.requires_grad = False
        
        weight_decay = self.config.weight_decay
        print(f"📊 Hyperparameters: lr={self.config.f_lr}, wd={weight_decay}, epochs={self.config.f_epochs}")
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.config.f_lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.student.ticket = False
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.result_dir, f'finetuned_model_best_{self.dataset_mode}.pth')
        
        for epoch in range(self.config.f_epochs):
            self.student.train()
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            meter_top1_train = meter.AverageMeter("Train Acc@1", ":6.2f")
            
            for images, targets in tqdm(self.train_loader, 
                                       desc=f"Epoch {epoch+1}/{self.config.f_epochs} [Train]", 
                                       ncols=100):
                images, targets = images.to(self.device), targets.to(self.device).float()
                
                optimizer.zero_grad()
                logits, _ = self.student(images)
                logits = logits.squeeze()
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                
                meter_loss.update(loss.item(), images.size(0))
                meter_top1_train.update(prec1, images.size(0))
            
            val_metrics = self.compute_metrics(
                self.val_loader, 
                description=f"Epoch_{epoch+1}_Val", 
                print_metrics=False, 
                save_confusion_matrix=False
            )
            val_acc = val_metrics['accuracy']
            
            print(f"Epoch {epoch+1}: Loss={meter_loss.avg:.4f}, "
                  f"Train Acc={meter_top1_train.avg:.2f}%, Val Acc={val_acc:.2f}%")
            
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  🎯 New best! Val Acc={best_val_acc:.2f}% → Saving model")
                torch.save(self.student.state_dict(), best_model_path)
        
        print(f"\n✅ Fine-tuning finished. Best Val Acc: {best_val_acc:.2f}%")
        
        if os.path.exists(best_model_path):
            self.student.load_state_dict(torch.load(best_model_path))
            print(f"✅ Loaded best model from {best_model_path}")
        else:
            print("⚠️  No best model saved, using last epoch model")

    def main(self):
        print(f"\n{'='*60}")
        print(f"شروع تست مدل: {self.dataset_mode}")
        print(f"{'='*60}\n")
        
        self.dataload()
        self.build_model()
        
        print("\n--- Testing BEFORE fine-tuning ---")
        initial_metrics = self.compute_metrics(self.test_loader, "Initial_Test")
        self.display_samples(initial_metrics['sample_info'], "Initial Test", num_samples=30)
        
        print("\n--- Starting fine-tuning ---")
        self.finetune()
        
        print("\n--- Testing AFTER fine-tuning ---")
        final_metrics = self.compute_metrics(self.test_loader, "Final_Test")
        self.display_samples(final_metrics['sample_info'], "Final Test", num_samples=30)


if __name__ == "__main__":
    config = Config()
    tester = Test(config)
    tester.main()
