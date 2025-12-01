import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from fractal_features import extract_zfrac, get_feature_dim


CACHE_DIR = "cache"


class KolektorSDD(Dataset):
    def __init__(self, root_dir, split='train', transform=None, use_zfrac=False,
                 grid_sizes=[1, 2, 4], train_ratio=0.7, val_ratio=0.15, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.use_zfrac = use_zfrac
        self.grid_sizes = grid_sizes
        self.classes = ['normal', 'defect']
        
        all_paths = []
        all_labels = []
        
        folders = sorted([d for d in os.listdir(root_dir) if d.startswith('kos')])
        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
            for f in os.listdir(folder_path):
                if f.endswith('.jpg'):
                    img_path = os.path.join(folder_path, f)
                    label_path = os.path.join(folder_path, f.replace('.jpg', '_label.bmp'))
                    if os.path.exists(label_path):
                        mask = np.array(Image.open(label_path))
                        label = 1 if np.any(mask > 0) else 0
                        all_paths.append(img_path)
                        all_labels.append(label)
        
        np.random.seed(seed)
        idx = np.random.permutation(len(all_paths))
        
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        
        if split == 'train':
            split_idx = idx[:n_train]
        elif split == 'val':
            split_idx = idx[n_train:n_train + n_val]
        else:
            split_idx = idx[n_train + n_val:]
        
        self.paths = [all_paths[i] for i in split_idx]
        self.labels = np.array([all_labels[i] for i in split_idx])
        self.split = split
        
        self.zfrac_features = None
        self.mean = None
        self.std = None
        
        if use_zfrac:
            self._load_or_extract_features()
    
    def _get_cache_path(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        gs_str = "_".join(map(str, self.grid_sizes))
        return os.path.join(CACHE_DIR, f"kolektor_zfrac_{self.split}_{gs_str}.npy")
    
    def _load_or_extract_features(self):
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"loading cached features from {cache_path}")
            self.zfrac_features = np.load(cache_path)
        else:
            print(f"extracting zfrac features for {self.split}...")
            feats = []
            for p in tqdm(self.paths):
                img = Image.open(p).convert('RGB')
                f = extract_zfrac(np.array(img), self.grid_sizes)
                feats.append(f)
            self.zfrac_features = np.stack(feats)
            np.save(cache_path, self.zfrac_features)
            print(f"saved to {cache_path}")
    
    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.use_zfrac:
            feat = self.zfrac_features[idx].copy()
            if self.mean is not None:
                feat = (feat - self.mean) / self.std
            return torch.tensor(feat, dtype=torch.float32), label
        else:
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
    
    @property
    def num_classes(self):
        return 2


class TomatoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, use_zfrac=False, 
                 grid_sizes=[1, 2, 4], train_ratio=0.7, val_ratio=0.15, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.use_zfrac = use_zfrac
        self.grid_sizes = grid_sizes
        
        self.classes = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        all_paths = []
        all_labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_paths.append(os.path.join(cls_dir, img_name))
                    all_labels.append(self.class_to_idx[cls])
        
        np.random.seed(seed)
        idx = np.random.permutation(len(all_paths))
        
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        
        if split == 'train':
            split_idx = idx[:n_train]
        elif split == 'val':
            split_idx = idx[n_train:n_train + n_val]
        else:
            split_idx = idx[n_train + n_val:]
        
        self.paths = [all_paths[i] for i in split_idx]
        self.labels = np.array([all_labels[i] for i in split_idx])
        self.split = split
        
        self.zfrac_features = None
        self.mean = None
        self.std = None
        
        if use_zfrac:
            self._load_or_extract_features()
    
    def _get_cache_path(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        gs_str = "_".join(map(str, self.grid_sizes))
        return os.path.join(CACHE_DIR, f"tomato_zfrac_{self.split}_{gs_str}.npy")
    
    def _load_or_extract_features(self):
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"loading cached features from {cache_path}")
            self.zfrac_features = np.load(cache_path)
        else:
            print(f"extracting zfrac features for {self.split}...")
            feats = []
            for p in tqdm(self.paths):
                img = Image.open(p).convert('RGB')
                f = extract_zfrac(np.array(img), self.grid_sizes)
                feats.append(f)
            self.zfrac_features = np.stack(feats)
            np.save(cache_path, self.zfrac_features)
            print(f"saved to {cache_path}")
    
    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.use_zfrac:
            feat = self.zfrac_features[idx].copy()
            if self.mean is not None:
                feat = (feat - self.mean) / self.std
            return torch.tensor(feat, dtype=torch.float32), label
        else:
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
    
    @property
    def num_classes(self):
        return len(self.classes)


class MagneticTileDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, use_zfrac=False,
                 grid_sizes=[1, 2, 4], train_ratio=0.7, val_ratio=0.15, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.use_zfrac = use_zfrac
        self.grid_sizes = grid_sizes
        
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('MT_')])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        all_paths = []
        all_labels = []
        
        for cls in self.classes:
            imgs_dir = os.path.join(root_dir, cls, 'Imgs')
            if not os.path.exists(imgs_dir):
                imgs_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(imgs_dir):
                if img_name.lower().endswith('.jpg'):
                    all_paths.append(os.path.join(imgs_dir, img_name))
                    all_labels.append(self.class_to_idx[cls])
        
        np.random.seed(seed)
        idx = np.random.permutation(len(all_paths))
        
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        
        if split == 'train':
            split_idx = idx[:n_train]
        elif split == 'val':
            split_idx = idx[n_train:n_train + n_val]
        else:
            split_idx = idx[n_train + n_val:]
        
        self.paths = [all_paths[i] for i in split_idx]
        self.labels = np.array([all_labels[i] for i in split_idx])
        self.split = split
        
        self.zfrac_features = None
        self.mean = None
        self.std = None
        
        if use_zfrac:
            self._load_or_extract_features()
    
    def _get_cache_path(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        gs_str = "_".join(map(str, self.grid_sizes))
        return os.path.join(CACHE_DIR, f"magnetic_zfrac_{self.split}_{gs_str}.npy")
    
    def _load_or_extract_features(self):
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"loading cached features from {cache_path}")
            self.zfrac_features = np.load(cache_path)
        else:
            print(f"extracting zfrac features for {self.split}...")
            feats = []
            for p in tqdm(self.paths):
                img = Image.open(p).convert('RGB')
                f = extract_zfrac(np.array(img), self.grid_sizes)
                feats.append(f)
            self.zfrac_features = np.stack(feats)
            np.save(cache_path, self.zfrac_features)
            print(f"saved to {cache_path}")
    
    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.use_zfrac:
            feat = self.zfrac_features[idx].copy()
            if self.mean is not None:
                feat = (feat - self.mean) / self.std
            return torch.tensor(feat, dtype=torch.float32), label
        else:
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
    
    @property
    def num_classes(self):
        return len(self.classes)


def get_transforms(train=True, size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def get_tomato_loaders(data_root, batch_size=32, use_zfrac=False, grid_sizes=[1, 2, 4], num_workers=0):
    train_tf = get_transforms(True) if not use_zfrac else None
    eval_tf = get_transforms(False) if not use_zfrac else None
    
    train_ds = TomatoDataset(data_root, 'train', train_tf, use_zfrac, grid_sizes)
    val_ds = TomatoDataset(data_root, 'val', eval_tf, use_zfrac, grid_sizes)
    test_ds = TomatoDataset(data_root, 'test', eval_tf, use_zfrac, grid_sizes)
    
    if use_zfrac:
        mean = np.mean(train_ds.zfrac_features, axis=0)
        std = np.std(train_ds.zfrac_features, axis=0)
        std[std == 0] = 1.0
        
        train_ds.set_normalization(mean, std)
        val_ds.set_normalization(mean, std)
        test_ds.set_normalization(mean, std)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=True if not use_zfrac else False)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True if not use_zfrac else False)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True if not use_zfrac else False)
    
    input_dim = get_feature_dim(grid_sizes) if use_zfrac else None
    
    return train_loader, val_loader, test_loader, train_ds.num_classes, input_dim


def get_kolektor_loaders(data_root, batch_size=32, use_zfrac=False, grid_sizes=[1, 2, 4], num_workers=0):
    train_tf = get_transforms(True) if not use_zfrac else None
    eval_tf = get_transforms(False) if not use_zfrac else None
    
    train_ds = KolektorSDD(data_root, 'train', train_tf, use_zfrac, grid_sizes)
    val_ds = KolektorSDD(data_root, 'val', eval_tf, use_zfrac, grid_sizes)
    test_ds = KolektorSDD(data_root, 'test', eval_tf, use_zfrac, grid_sizes)
    
    if use_zfrac:
        mean = np.mean(train_ds.zfrac_features, axis=0)
        std = np.std(train_ds.zfrac_features, axis=0)
        std[std == 0] = 1.0
        
        train_ds.set_normalization(mean, std)
        val_ds.set_normalization(mean, std)
        test_ds.set_normalization(mean, std)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if not use_zfrac else False)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True if not use_zfrac else False)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True if not use_zfrac else False)
    
    input_dim = get_feature_dim(grid_sizes) if use_zfrac else None
    
    return train_loader, val_loader, test_loader, train_ds.num_classes, input_dim


def get_magnetic_loaders(data_root, batch_size=32, use_zfrac=False, grid_sizes=[1, 2, 4], num_workers=0):
    train_tf = get_transforms(True) if not use_zfrac else None
    eval_tf = get_transforms(False) if not use_zfrac else None
    
    train_ds = MagneticTileDataset(data_root, 'train', train_tf, use_zfrac, grid_sizes)
    val_ds = MagneticTileDataset(data_root, 'val', eval_tf, use_zfrac, grid_sizes)
    test_ds = MagneticTileDataset(data_root, 'test', eval_tf, use_zfrac, grid_sizes)
    
    if use_zfrac:
        mean = np.mean(train_ds.zfrac_features, axis=0)
        std = np.std(train_ds.zfrac_features, axis=0)
        std[std == 0] = 1.0
        
        train_ds.set_normalization(mean, std)
        val_ds.set_normalization(mean, std)
        test_ds.set_normalization(mean, std)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if not use_zfrac else False)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True if not use_zfrac else False)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True if not use_zfrac else False)
    
    input_dim = get_feature_dim(grid_sizes) if use_zfrac else None
    
    return train_loader, val_loader, test_loader, train_ds.num_classes, input_dim
