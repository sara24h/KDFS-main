import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
import numpy as np
import os
import random
from pathlib import Path
from torchvision import transforms

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_seed(42)

def worker_init_fn(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class UADFVDataset(Dataset):
    def __init__(self, root_dir, num_frames=32, image_size=256,
                 transform=None, sampling_strategy='uniform',
                 split='train', split_ratio=(0.8, 0.1, 0.1), seed=42):
        
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.sampling_strategy = sampling_strategy
        self.split = split
        self.seed = seed

        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
            else:  # val / test
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        # بارگذاری و تقسیم ویدیوها
        self.video_list = self._load_and_split(split_ratio)
        print(f"[{split.upper()}] {len(self.video_list)} videos loaded.")

    def _load_and_split(self, split_ratio):
        video_list = []

        # Fake → label 0
        for p in sorted((self.root_dir / 'fake').glob('*.mp4')):
            if not p.name.startswith('.'):
                video_list.append((str(p), 0))

        # Real → label 1
        for p in sorted((self.root_dir / 'real').glob('*.mp4')):
            if not p.name.startswith('.'):
                video_list.append((str(p), 1))

        # Shuffle ثابت
        rng = random.Random(self.seed)
        rng.shuffle(video_list)

        total = len(video_list)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])

        if self.split == 'train':
            return video_list[:train_end]
        elif self.split == 'val':
            return video_list[train_end:val_end]
        elif self.split == 'test':
            return video_list[val_end:]
        else:
            raise ValueError("split must be train/val/test")

    def sample_frames(self, total_frames: int):
        if total_frames <= self.num_frames:
            idxs = np.random.choice(total_frames, self.num_frames, replace=True)
            return sorted(idxs.tolist())

        if self.sampling_strategy == 'uniform':
            return np.linspace(0, total_frames-1, self.num_frames, dtype=int).tolist()
        elif self.sampling_strategy == 'random':
            idxs = np.random.choice(total_frames, self.num_frames, replace=False)
            return sorted(idxs.tolist())
        elif self.sampling_strategy == 'first':
            return list(range(self.num_frames))
        else:
            raise ValueError("sampling_strategy: uniform / random / first")

    def load_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self.sample_frames(total)
        frames = []

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    frame = self.transform(frame)
                except Exception as e:
                    print(f"Transform error on frame from {path}: {e}")
                    frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(frame)
            else:
                # fallback
                fallback = frames[-1].clone() if frames else torch.zeros(3, self.image_size, self.image_size)
                frames.append(fallback)

        cap.release()
        return torch.stack(frames)  # [T, C, H, W]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        path, label = self.video_list[idx]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = self.seed + worker_info.id * 100000 + idx
        else:
            seed = self.seed + idx

        r_state = random.getstate()
        np_state = np.random.get_state()

        random.seed(seed)
        np.random.seed(seed)

        try:
            frames = self.load_video(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        finally:
            random.setstate(r_state)
            np.random.set_state(np_state)

        return frames, torch.tensor(label, dtype=torch.float32)


def create_uadfv_dataloaders(
    root_dir,
    num_frames=16,
    image_size=256,
    train_batch_size=8,
    eval_batch_size=16,
    num_workers=4,
    pin_memory=True,          
    ddp=False,
    sampling_strategy='uniform',
    seed=42
):
    train_ds = UADFVDataset(root_dir, num_frames, image_size,
                            sampling_strategy=sampling_strategy,
                            split='train', seed=seed)
    val_ds   = UADFVDataset(root_dir, num_frames, image_size,
                            sampling_strategy=sampling_strategy,
                            split='val', seed=seed)
    test_ds  = UADFVDataset(root_dir, num_frames, image_size,
                            sampling_strategy=sampling_strategy,
                            split='test', seed=seed)

    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler   = DistributedSampler(val_ds, shuffle=False)
        test_sampler  = DistributedSampler(test_ds, shuffle=False)
        shuffle = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle = True

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds,
                              batch_size=train_batch_size,
                              shuffle=shuffle,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              drop_last=True,
                              worker_init_fn=worker_init_fn,
                              generator=g)

    val_loader = DataLoader(val_ds,
                            batch_size=eval_batch_size,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            worker_init_fn=worker_init_fn)

    test_loader = DataLoader(test_ds,
                             batch_size=eval_batch_size,
                             shuffle=False,
                             sampler=test_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root_dir = "/kaggle/input/uadfv-dataset/UADFV"   

    train_loader, val_loader, test_loader = create_uadfv_dataloaders(
        root_dir=root_dir,
        num_frames=16,
        image_size=256,
        train_batch_size=4,
        eval_batch_size=8,
        num_workers=4,
        pin_memory=True,
        ddp=False,
        sampling_strategy='uniform' 
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    for videos, labels in train_loader:
        print("Batch shape :", videos.shape)        # [B, T, C, H, W]
        print("Labels      :", labels.tolist())
        print("First video mean pixel:", videos[0].mean().item())
        break
