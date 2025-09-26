import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode
        self.result_dir = args.result_dir
        self.new_dataset_dir = getattr(args, 'new_dataset_dir', None)  # New dataset directory

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")
            
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None  # Loader for new test dataset
        self.student = None

    def dataload(self):
        print("==> Loading datasets...")
        
        image_size = (256, 256)
        mean_10k = [0.5212, 0.4260, 0.3811]
        std_10k = [0.2486, 0.2238, 0.2211]

        transform_train_10k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_10k, std=std_10k),
        ])

        transform_val_test_10k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_10k, std=std_10k),
        ])

        params = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'ddp': False
        }
        
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

        dataset_manager = Dataset_selector(**params)

        print("Overriding transforms to use consistent 10k normalization stats for all datasets.")
        dataset_manager.loader_train.dataset.transform = transform_train_10k
        dataset_manager.loader_val.dataset.transform = transform_val_test_10k
        dataset_manager.loader_test.dataset.transform = transform_val_test_10k

        self.train_loader = dataset_manager.loader_train
        self.val_loader = dataset_manager.loader_val
        self.test_loader = dataset_manager.loader_test
        
        print(f"All loaders for '{self.dataset_mode}' are now configured with 10k normalization.")

        # Load new test dataset if provided
        if self.new_dataset_dir:
            print("==> Loading new test dataset...")
            new_params = {
                'dataset_mode': 'new_test',
                'eval_batch_size': self.test_batch_size,
                'num_workers': self.num_workers,
                'pin_memory': self.pin_memory,
                'new_test_csv': os.path.join(self.new_dataset_dir, 'test.csv'),
                'new_test_root_dir': self.new_dataset_dir
            }
            new_dataset_manager = Dataset_selector(**new_params)
            new_dataset_manager.loader_test.dataset.transform = transform_val_test_10k
            self.new_test_loader = new_dataset_manager.loader_test
            print(f"New test dataset loader configured with 10k normalization.")

    def build_model(self):
        print("==> Building student model...")
        self.student = ResNet_50_sparse_hardfakevsreal()
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            
        print(f"Loading pre-trained weights from: {self.sparsed_student_ckpt_path}")
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt_student.get("student", ckpt_student)
        
        self.student.load_state_dict(state_dict, strict=False)
        
        # Add dropout before the fc layer
        self.student.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            self.student.fc
        )
        
        self.student.to(self.device)
        print(f"Model loaded on {self.device}")

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
            print(f"[{description}] Overall Metrics:")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Specificity: {specificity_real:.4f}")
            
            print(f"\n[{description}] Per-Class Metrics:")
            print(f"Class Real (0):")
            print(f"  Precision: {precision_per_class[0]:.4f}")
            print(f"  Recall: {recall_per_class[0]:.4f}")
            print(f"  Specificity: {specificity_real:.4f}")
            print(f"Class Fake (1):")
            print(f"  Precision: {precision_per_class[1]:.4f}")
            print(f"  Recall: {recall_per_class[1]:.4f}")
            print(f"  Specificity: {specificity_fake:.4f}")
        
        cm = confusion_matrix(all_targets, all_preds)
        classes = ['Real', 'Fake']
        
        if save_confusion_matrix:
            print(f"\n[{description}] Confusion Matrix:")
            print(f"{'':>10} {'Predicted Real':>15} {'Predicted Fake':>15}")
            print(f"{'Actual Real':>10} {cm[0,0]:>15} {cm[0,1]:>15}")
            print(f"{'Actual Fake':>10} {cm[1,0]:>15} {cm[1,1]:>15}")
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix - {description}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            sanitized_description = description.lower().replace(" ", "_").replace("/", "_")
            plot_path = os.path.join(self.result_dir, f'confusion_matrix_{sanitized_description}.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
            print(f"Confusion matrix saved to: {plot_path}")
        
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
        print(f"\n[{description}] Displaying first {num_samples} test samples:")
        print(f"{'Sample ID':<50} {'True Label':<12} {'Predicted Label':<12}")
        print("-" * 80)
        for i, sample in enumerate(sample_info[:num_samples]):
            true_label = 'Real' if sample['true_label'] == 0 else 'Fake'
            pred_label = 'Real' if sample['pred_label'] == 0 else 'Fake'
            print(f"{sample['id']:<50} {true_label:<12} {pred_label:<12}")

    def finetune(self):
        print("==> Fine-tuning using FEATURE EXTRACTOR strategy on 'fc' and 'layer4'...")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        for name, param in self.student.named_parameters():
            if 'fc' in name or 'layer4' in name:
                param.requires_grad = True
                print(f"Unfreezing for training: {name}")
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.args.f_lr,
            weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.student.ticket = False
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.result_dir, f'finetuned_model_best_{self.dataset_mode}.pth')

        for epoch in range(self.args.f_epochs):
            self.student.train()
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            meter_top1_train = meter.AverageMeter("Train Acc@1", ":6.2f")
            
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.f_epochs} [Train]", ncols=100):
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

            # Compute validation metrics
            val_metrics = self.compute_metrics(self.val_loader, description=f"Epoch_{epoch+1}_{self.args.f_epochs}_Val", print_metrics=False, save_confusion_matrix=False)
            val_acc = val_metrics['accuracy']
            
            # Print train and validation metrics for the epoch
            print(f"Epoch {epoch+1}: Train Loss: {meter_loss.avg:.4f}, Train Acc: {meter_top1_train.avg:.2f}%, Val Acc: {val_acc:.2f}%")

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best model found with Val Acc: {best_val_acc:.2f}%. Saving to {best_model_path}")
                torch.save(self.student.state_dict(), best_model_path)
        
        print(f"\nFine-tuning finished. Loading best model with Val Acc: {best_val_acc:.2f}%")
        if os.path.exists(best_model_path):
            self.student.load_state_dict(torch.load(best_model_path))
        else:
            print("Warning: No best model was saved. The model from the last epoch will be used for testing.")
        
        # Compute and print final test metrics after fine-tuning
        final_test_metrics = self.compute_metrics(self.test_loader, description="Final_Test", print_metrics=True, save_confusion_matrix=True)
        print(f"\nFinal Test Metrics after Fine-tuning:")
        print(f"Accuracy: {final_test_metrics['accuracy']:.2f}%")
        print(f"Precision: {final_test_metrics['precision']:.4f}")
        print(f"Recall: {final_test_metrics['recall']:.4f}")
        print(f"Specificity: {final_test_metrics['specificity']:.4f}")
        print(f"\nPer-Class Metrics:")
        print(f"Class Real (0):")
        print(f"  Precision: {final_test_metrics['precision_per_class'][0]:.4f}")
        print(f"  Recall: {final_test_metrics['recall_per_class'][0]:.4f}")
        print(f"  Specificity: {final_test_metrics['specificity_per_class'][0]:.4f}")
        print(f"Class Fake (1):")
        print(f"  Precision: {final_test_metrics['precision_per_class'][1]:.4f}")
        print(f"  Recall: {final_test_metrics['recall_per_class'][1]:.4f}")
        print(f"  Specificity: {final_test_metrics['specificity_per_class'][1]:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"{'':>10} {'Predicted Real':>15} {'Predicted Fake':>15}")
        print(f"{'Actual Real':>10} {final_test_metrics['confusion_matrix'][0,0]:>15} {final_test_metrics['confusion_matrix'][0,1]:>15}")
        print(f"{'Actual Fake':>10} {final_test_metrics['confusion_matrix'][1,0]:>15} {final_test_metrics['confusion_matrix'][1,1]:>15}")

    def main(self):
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        self.dataload()
        self.build_model()
        
        print("\n--- Testing BEFORE fine-tuning ---")
        initial_metrics = self.compute_metrics(self.test_loader, "Initial_Test")
        self.display_samples(initial_metrics['sample_info'], "Initial Test", num_samples=30)
        
        print("\n--- Starting fine-tuning ---")
        self.finetune()
        
        print("\n--- Testing AFTER fine-tuning with best model ---")
        final_metrics = self.compute_metrics(self.test_loader, "Final_Test", print_metrics=False)
        self.display_samples(final_metrics['sample_info'], "Final Test", num_samples=30)
        
        if self.new_test_loader:
            print("\n--- Testing on NEW dataset ---")
            new_metrics = self.compute_metrics(self.new_test_loader, "New_Dataset_Test")
            self.display_samples(new_metrics['sample_info'], "New Dataset Test", num_samples=30)
