import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from fractal_features import extract_zfrac, get_feature_dim


# Constants
CACHE_DIR = "cache"
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SEED = 42
DEFAULT_GRID_SIZES = [1, 2, 4]
DEFAULT_MAX_SAMPLES = 10000


def split_dataset_indices(total_samples, train_ratio, val_ratio, seed):
    """
    Split dataset indices into train, validation, and test sets.
    
    Args:
        total_samples: Total number of samples in the dataset
        train_ratio: Fraction of data to use for training
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', and 'test' indices
    """
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(total_samples)
    
    num_train = int(total_samples * train_ratio)
    num_val = int(total_samples * val_ratio)
    
    split_indices = {
        'train': shuffled_indices[:num_train],
        'val': shuffled_indices[num_train:num_train + num_val],
        'test': shuffled_indices[num_train + num_val:]
    }
    
    return split_indices


def get_split_indices_for_split(split_indices_dict, split_name):
    """
    Get indices for a specific split (train, val, or test).
    
    Args:
        split_indices_dict: Dictionary returned by split_dataset_indices
        split_name: Name of the split ('train', 'val', or 'test')
    
    Returns:
        Array of indices for the requested split
    """
    return split_indices_dict[split_name]


class KolektorSDD(Dataset):
    """
    Dataset class for Kolektor Surface Defect Dataset.
    
    This dataset contains images of surface defects. Labels are determined by
    checking if corresponding label mask files contain any non-zero pixels.
    """
    
    def __init__(self, root_dir, split='train', transform=None, use_zfrac=False,
                 grid_sizes=DEFAULT_GRID_SIZES, train_ratio=DEFAULT_TRAIN_RATIO, 
                 val_ratio=DEFAULT_VAL_RATIO, seed=DEFAULT_SEED):
        self.root_dir = root_dir
        self.transform = transform
        self.use_zfrac = use_zfrac
        self.grid_sizes = grid_sizes
        self.classes = ['normal', 'defect']
        self.split = split
        
        # Collect all image paths and labels from dataset folders
        all_image_paths = []
        all_image_labels = []
        
        # Find all folders that start with 'kos' (Kolektor dataset naming convention)
        dataset_folders = []
        for directory_name in os.listdir(root_dir):
            if directory_name.startswith('kos'):
                dataset_folders.append(directory_name)
        dataset_folders = sorted(dataset_folders)
        
        # Process each folder to find images and their corresponding labels
        for folder_name in dataset_folders:
            folder_path = os.path.join(root_dir, folder_name)
            
            for filename in os.listdir(folder_path):
                # Only process JPG images
                if filename.endswith('.jpg'):
                    image_path = os.path.join(folder_path, filename)
                    # Label file has same name but with '_label.bmp' extension
                    label_filename = filename.replace('.jpg', '_label.bmp')
                    label_path = os.path.join(folder_path, label_filename)
                    
                    # Only include images that have corresponding label files
                    if os.path.exists(label_path):
                        # Label is 1 if mask has any defects (non-zero pixels), else 0
                        label_mask = np.array(Image.open(label_path))
                        has_defect = np.any(label_mask > 0)
                        label = 1 if has_defect else 0
                        
                        all_image_paths.append(image_path)
                        all_image_labels.append(label)
        
        # Split dataset into train/val/test
        split_indices = split_dataset_indices(
            len(all_image_paths), train_ratio, val_ratio, seed
        )
        split_indices_for_current_split = get_split_indices_for_split(split_indices, split)
        
        # Store paths and labels for this split
        self.paths = [all_image_paths[index] for index in split_indices_for_current_split]
        self.labels = np.array([all_image_labels[index] for index in split_indices_for_current_split])
        
        # Initialize feature storage (will be populated if use_zfrac is True)
        self.zfrac_features = None
        self.mean = None
        self.std = None
        
        if use_zfrac:
            self._load_or_extract_features()
    
    def _get_cache_path(self):
        """
        Generate the cache file path for storing/loading extracted features.
        
        Returns:
            Path to the cache file for this dataset's features
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        grid_sizes_string = "_".join(map(str, self.grid_sizes))
        cache_filename = f"kolektor_zfrac_{self.split}_{grid_sizes_string}.npy"
        return os.path.join(CACHE_DIR, cache_filename)
    
    def _load_or_extract_features(self):
        """
        Load cached features if available, otherwise extract and cache them.
        
        Feature extraction can be slow, so we cache results to disk for faster
        subsequent loads.
        """
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            self.zfrac_features = np.load(cache_path)
        else:
            print(f"Extracting zfrac features for {self.split} split...")
            feature_list = []
            
            for image_path in tqdm(self.paths):
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                features = extract_zfrac(image_array, self.grid_sizes)
                feature_list.append(features)
            
            self.zfrac_features = np.stack(feature_list)
            np.save(cache_path, self.zfrac_features)
            print(f"Saved features to {cache_path}")
    
    def set_normalization(self, mean, std):
        """
        Set normalization parameters for features.
        
        These should be computed from the training set and applied consistently
        to validation and test sets.
        
        Args:
            mean: Mean values for each feature dimension
            std: Standard deviation values for each feature dimension
        """
        self.mean = mean
        self.std = std
    
    def __len__(self):
        """Return the number of samples in this dataset split."""
        return len(self.paths)
    
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
        
        Returns:
            Tuple of (features_or_image, label)
        """
        label = self.labels[index]
        
        if self.use_zfrac:
            # Return pre-extracted fractal features
            features = self.zfrac_features[index].copy()
            
            # Apply normalization if parameters are set
            if self.mean is not None:
                features = (features - self.mean) / self.std
            
            return torch.tensor(features, dtype=torch.float32), label
        else:
            # Return raw image with transforms applied
            image = Image.open(self.paths[index]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    @property
    def num_classes(self):
        """Return the number of classes in this dataset."""
        return 2


class TomatoDataset(Dataset):
    """
    Dataset class for Tomato Leaf Disease Dataset.
    
    This dataset contains images organized in folders by disease class.
    Each folder name represents a different class.
    """
    
    def __init__(self, root_dir, split='train', transform=None, use_zfrac=False, 
                 grid_sizes=DEFAULT_GRID_SIZES, train_ratio=DEFAULT_TRAIN_RATIO, 
                 val_ratio=DEFAULT_VAL_RATIO, seed=DEFAULT_SEED):
        self.root_dir = root_dir
        self.transform = transform
        self.use_zfrac = use_zfrac
        self.grid_sizes = grid_sizes
        self.split = split
        
        # Find all class directories (each folder is a class)
        class_names = []
        for directory_name in os.listdir(root_dir):
            directory_path = os.path.join(root_dir, directory_name)
            if os.path.isdir(directory_path):
                class_names.append(directory_name)
        self.classes = sorted(class_names)
        
        # Create mapping from class name to index
        self.class_to_idx = {}
        for class_index, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = class_index
        
        # Collect all image paths and labels
        all_image_paths = []
        all_image_labels = []
        
        for class_name in self.classes:
            class_directory = os.path.join(root_dir, class_name)
            
            for image_filename in os.listdir(class_directory):
                # Accept common image formats
                if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_directory, image_filename)
                    class_index = self.class_to_idx[class_name]
                    
                    all_image_paths.append(image_path)
                    all_image_labels.append(class_index)
        
        # Split dataset into train/val/test
        split_indices = split_dataset_indices(
            len(all_image_paths), train_ratio, val_ratio, seed
        )
        split_indices_for_current_split = get_split_indices_for_split(split_indices, split)
        
        # Store paths and labels for this split
        self.paths = [all_image_paths[index] for index in split_indices_for_current_split]
        self.labels = np.array([all_image_labels[index] for index in split_indices_for_current_split])
        
        # Initialize feature storage
        self.zfrac_features = None
        self.mean = None
        self.std = None
        
        if use_zfrac:
            self._load_or_extract_features()
    
    def _get_cache_path(self):
        """Generate the cache file path for storing/loading extracted features."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        grid_sizes_string = "_".join(map(str, self.grid_sizes))
        cache_filename = f"tomato_zfrac_{self.split}_{grid_sizes_string}.npy"
        return os.path.join(CACHE_DIR, cache_filename)
    
    def _load_or_extract_features(self):
        """Load cached features if available, otherwise extract and cache them."""
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            self.zfrac_features = np.load(cache_path)
        else:
            print(f"Extracting zfrac features for {self.split} split...")
            feature_list = []
            
            for image_path in tqdm(self.paths):
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                features = extract_zfrac(image_array, self.grid_sizes)
                feature_list.append(features)
            
            self.zfrac_features = np.stack(feature_list)
            np.save(cache_path, self.zfrac_features)
            print(f"Saved features to {cache_path}")
    
    def set_normalization(self, mean, std):
        """Set normalization parameters for features."""
        self.mean = mean
        self.std = std
    
    def __len__(self):
        """Return the number of samples in this dataset split."""
        return len(self.paths)
    
    def __getitem__(self, index):
        """Get a single sample from the dataset."""
        label = self.labels[index]
        
        if self.use_zfrac:
            features = self.zfrac_features[index].copy()
            if self.mean is not None:
                features = (features - self.mean) / self.std
            return torch.tensor(features, dtype=torch.float32), label
        else:
            image = Image.open(self.paths[index]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    @property
    def num_classes(self):
        """Return the number of classes in this dataset."""
        return len(self.classes)


class MagneticTileDataset(Dataset):
    """
    Dataset class for Magnetic Tile Defect Dataset.
    
    This dataset contains images of magnetic tiles with various defect types.
    Classes are organized in folders starting with 'MT_'. Images may be in
    an 'Imgs' subdirectory or directly in the class folder.
    """
    
    def __init__(self, root_dir, split='train', transform=None, use_zfrac=False,
                 grid_sizes=DEFAULT_GRID_SIZES, train_ratio=DEFAULT_TRAIN_RATIO, 
                 val_ratio=DEFAULT_VAL_RATIO, seed=DEFAULT_SEED):
        self.root_dir = root_dir
        self.transform = transform
        self.use_zfrac = use_zfrac
        self.grid_sizes = grid_sizes
        self.split = split
        
        # Find all class directories that start with 'MT_'
        class_names = []
        for directory_name in os.listdir(root_dir):
            directory_path = os.path.join(root_dir, directory_name)
            if os.path.isdir(directory_path) and directory_name.startswith('MT_'):
                class_names.append(directory_name)
        self.classes = sorted(class_names)
        
        # Create mapping from class name to index
        self.class_to_idx = {}
        for class_index, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = class_index
        
        # Collect all image paths and labels
        all_image_paths = []
        all_image_labels = []
        
        for class_name in self.classes:
            # Images may be in 'Imgs' subdirectory or directly in class folder
            images_directory = os.path.join(root_dir, class_name, 'Imgs')
            if not os.path.exists(images_directory):
                images_directory = os.path.join(root_dir, class_name)
            
            for image_filename in os.listdir(images_directory):
                if image_filename.lower().endswith('.jpg'):
                    image_path = os.path.join(images_directory, image_filename)
                    class_index = self.class_to_idx[class_name]
                    
                    all_image_paths.append(image_path)
                    all_image_labels.append(class_index)
        
        # Split dataset into train/val/test
        split_indices = split_dataset_indices(
            len(all_image_paths), train_ratio, val_ratio, seed
        )
        split_indices_for_current_split = get_split_indices_for_split(split_indices, split)
        
        # Store paths and labels for this split
        self.paths = [all_image_paths[index] for index in split_indices_for_current_split]
        self.labels = np.array([all_image_labels[index] for index in split_indices_for_current_split])
        
        # Initialize feature storage
        self.zfrac_features = None
        self.mean = None
        self.std = None
        
        if use_zfrac:
            self._load_or_extract_features()
    
    def _get_cache_path(self):
        """Generate the cache file path for storing/loading extracted features."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        grid_sizes_string = "_".join(map(str, self.grid_sizes))
        cache_filename = f"magnetic_zfrac_{self.split}_{grid_sizes_string}.npy"
        return os.path.join(CACHE_DIR, cache_filename)
    
    def _load_or_extract_features(self):
        """Load cached features if available, otherwise extract and cache them."""
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            self.zfrac_features = np.load(cache_path)
        else:
            print(f"Extracting zfrac features for {self.split} split...")
            feature_list = []
            
            for image_path in tqdm(self.paths):
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                features = extract_zfrac(image_array, self.grid_sizes)
                feature_list.append(features)
            
            self.zfrac_features = np.stack(feature_list)
            np.save(cache_path, self.zfrac_features)
            print(f"Saved features to {cache_path}")
    
    def set_normalization(self, mean, std):
        """Set normalization parameters for features."""
        self.mean = mean
        self.std = std
    
    def __len__(self):
        """Return the number of samples in this dataset split."""
        return len(self.paths)
    
    def __getitem__(self, index):
        """Get a single sample from the dataset."""
        label = self.labels[index]
        
        if self.use_zfrac:
            features = self.zfrac_features[index].copy()
            if self.mean is not None:
                features = (features - self.mean) / self.std
            return torch.tensor(features, dtype=torch.float32), label
        else:
            image = Image.open(self.paths[index]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    @property
    def num_classes(self):
        """Return the number of classes in this dataset."""
        return len(self.classes)


class SurfaceCracksDataset(Dataset):
    """
    Dataset class for Surface Cracks Dataset.
    
    This is a binary classification dataset with 'Negative' (no crack) and
    'Positive' (crack) classes. The dataset can be very large, so we limit
    the number of samples per class to manage memory and computation time.
    """
    
    def __init__(self, root_dir, split='train', transform=None, use_zfrac=False,
                 grid_sizes=DEFAULT_GRID_SIZES, train_ratio=DEFAULT_TRAIN_RATIO, 
                 val_ratio=DEFAULT_VAL_RATIO, seed=DEFAULT_SEED, max_samples=DEFAULT_MAX_SAMPLES):
        self.root_dir = root_dir
        self.transform = transform
        self.use_zfrac = use_zfrac
        self.grid_sizes = grid_sizes
        self.split = split
        self.classes = ['Negative', 'Positive']
        self.class_to_idx = {'Negative': 0, 'Positive': 1}
        self.max_samples = max_samples
        
        # Collect images from both classes
        all_image_paths = []
        all_image_labels = []
        
        # Use different random seeds for each class to ensure different selections
        # This prevents the same images from being selected across classes
        SEED_OFFSET_PER_CLASS = 1000
        
        for class_index, class_name in enumerate(self.classes):
            class_directory = os.path.join(root_dir, class_name)
            
            if os.path.exists(class_directory):
                class_image_paths = []
                
                for image_filename in os.listdir(class_directory):
                    if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(class_directory, image_filename)
                        class_image_paths.append(image_path)
                
                # Limit samples per class to max_samples/2 to keep total around max_samples
                # This ensures balanced classes while managing dataset size
                samples_per_class = max_samples // 2
                if len(class_image_paths) > samples_per_class:
                    # Use different seed for each class to get different random samples
                    class_specific_seed = seed + class_index * SEED_OFFSET_PER_CLASS
                    np.random.seed(class_specific_seed)
                    class_image_paths = np.random.choice(
                        class_image_paths, samples_per_class, replace=False
                    ).tolist()
                
                all_image_paths.extend(class_image_paths)
                # Create label list matching the number of images selected
                class_label = self.class_to_idx[class_name]
                all_image_labels.extend([class_label] * len(class_image_paths))
        
        # Split dataset into train/val/test
        split_indices = split_dataset_indices(
            len(all_image_paths), train_ratio, val_ratio, seed
        )
        split_indices_for_current_split = get_split_indices_for_split(split_indices, split)
        
        # Store paths and labels for this split
        self.paths = [all_image_paths[index] for index in split_indices_for_current_split]
        self.labels = np.array([all_image_labels[index] for index in split_indices_for_current_split])
        
        # Initialize feature storage
        self.zfrac_features = None
        self.mean = None
        self.std = None
        
        if use_zfrac:
            self._load_or_extract_features()
    
    def _get_cache_path(self):
        """Generate the cache file path for storing/loading extracted features."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        grid_sizes_string = "_".join(map(str, self.grid_sizes))
        cache_filename = f"surface_cracks_zfrac_{self.split}_{grid_sizes_string}.npy"
        return os.path.join(CACHE_DIR, cache_filename)
    
    def _load_or_extract_features(self):
        """Load cached features if available, otherwise extract and cache them."""
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            self.zfrac_features = np.load(cache_path)
        else:
            print(f"Extracting zfrac features for {self.split} split...")
            feature_list = []
            
            for image_path in tqdm(self.paths):
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                features = extract_zfrac(image_array, self.grid_sizes)
                feature_list.append(features)
            
            self.zfrac_features = np.stack(feature_list)
            np.save(cache_path, self.zfrac_features)
            print(f"Saved features to {cache_path}")
    
    def set_normalization(self, mean, std):
        """Set normalization parameters for features."""
        self.mean = mean
        self.std = std
    
    def __len__(self):
        """Return the number of samples in this dataset split."""
        return len(self.paths)
    
    def __getitem__(self, index):
        """Get a single sample from the dataset."""
        label = self.labels[index]
        
        if self.use_zfrac:
            features = self.zfrac_features[index].copy()
            if self.mean is not None:
                features = (features - self.mean) / self.std
            return torch.tensor(features, dtype=torch.float32), label
        else:
            image = Image.open(self.paths[index]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    @property
    def num_classes(self):
        """Return the number of classes in this dataset."""
        return 2


class NEUMetalSurfaceDataset(Dataset):
    """
    Dataset class for NEU Metal Surface Defects Dataset.
    
    This dataset is pre-split into train/valid/test directories. Unlike other
    datasets, we don't need to split it ourselves - we just load from the
    appropriate directory.
    """
    
    def __init__(self, root_dir, split='train', transform=None, use_zfrac=False,
                 grid_sizes=DEFAULT_GRID_SIZES, seed=DEFAULT_SEED):
        self.root_dir = root_dir
        self.transform = transform
        self.use_zfrac = use_zfrac
        self.grid_sizes = grid_sizes
        self.split = split
        
        # Map our split names to dataset directory names
        # The dataset uses 'valid' instead of 'val'
        split_name_mapping = {'train': 'train', 'val': 'valid', 'test': 'test'}
        dataset_split_name = split_name_mapping.get(split, 'train')
        split_directory_path = os.path.join(root_dir, dataset_split_name)
        
        # Find all class directories in the split folder
        class_names = []
        for directory_name in os.listdir(split_directory_path):
            directory_path = os.path.join(split_directory_path, directory_name)
            if os.path.isdir(directory_path):
                class_names.append(directory_name)
        self.classes = sorted(class_names)
        
        # Create mapping from class name to index
        self.class_to_idx = {}
        for class_index, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = class_index
        
        # Collect all image paths and labels from the split directory
        all_image_paths = []
        all_image_labels = []
        
        for class_name in self.classes:
            class_directory = os.path.join(split_directory_path, class_name)
            
            for image_filename in os.listdir(class_directory):
                # Accept common image formats including BMP
                if image_filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_directory, image_filename)
                    class_index = self.class_to_idx[class_name]
                    
                    all_image_paths.append(image_path)
                    all_image_labels.append(class_index)
        
        # This dataset is already split, so we use all images from the split directory
        self.paths = all_image_paths
        self.labels = np.array(all_image_labels)
        
        # Initialize feature storage
        self.zfrac_features = None
        self.mean = None
        self.std = None
        
        if use_zfrac:
            self._load_or_extract_features()
    
    def _get_cache_path(self):
        """Generate the cache file path for storing/loading extracted features."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        grid_sizes_string = "_".join(map(str, self.grid_sizes))
        cache_filename = f"neu_metal_zfrac_{self.split}_{grid_sizes_string}.npy"
        return os.path.join(CACHE_DIR, cache_filename)
    
    def _load_or_extract_features(self):
        """Load cached features if available, otherwise extract and cache them."""
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            self.zfrac_features = np.load(cache_path)
        else:
            print(f"Extracting zfrac features for {self.split} split...")
            feature_list = []
            
            for image_path in tqdm(self.paths):
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                features = extract_zfrac(image_array, self.grid_sizes)
                feature_list.append(features)
            
            self.zfrac_features = np.stack(feature_list)
            np.save(cache_path, self.zfrac_features)
            print(f"Saved features to {cache_path}")
    
    def set_normalization(self, mean, std):
        """Set normalization parameters for features."""
        self.mean = mean
        self.std = std
    
    def __len__(self):
        """Return the number of samples in this dataset split."""
        return len(self.paths)
    
    def __getitem__(self, index):
        """Get a single sample from the dataset."""
        label = self.labels[index]
        
        if self.use_zfrac:
            features = self.zfrac_features[index].copy()
            if self.mean is not None:
                features = (features - self.mean) / self.std
            return torch.tensor(features, dtype=torch.float32), label
        else:
            image = Image.open(self.paths[index]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    @property
    def num_classes(self):
        """Return the number of classes in this dataset."""
        return len(self.classes)


# ImageNet normalization constants (used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_ROTATION_DEGREES = 10
DEFAULT_COLOR_JITTER = 0.2


def get_transforms(is_training=True, image_size=DEFAULT_IMAGE_SIZE):
    """
    Get image transformation pipeline for training or evaluation.
    
    Training transforms include data augmentation (random flips, rotations, color jitter)
    to improve model generalization. Evaluation transforms only resize and normalize.
    
    Args:
        is_training: If True, apply data augmentation transforms
        image_size: Target size for resizing images
    
    Returns:
        A Compose object with the appropriate transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(DEFAULT_ROTATION_DEGREES),
            transforms.ColorJitter(DEFAULT_COLOR_JITTER, DEFAULT_COLOR_JITTER),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])


def compute_normalization_statistics(dataset):
    """
    Compute mean and standard deviation for feature normalization.
    
    These statistics are computed from the training set and applied to
    all splits to ensure consistent normalization.
    
    Args:
        dataset: Dataset object with zfrac_features attribute
    
    Returns:
        Tuple of (mean, std) arrays
    """
    mean = np.mean(dataset.zfrac_features, axis=0)
    std = np.std(dataset.zfrac_features, axis=0)
    # Prevent division by zero for constant features
    std[std == 0] = 1.0
    return mean, std


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, 
                        use_zfrac, num_workers=0):
    """
    Create DataLoader objects for train, validation, and test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        use_zfrac: Whether using fractal features (affects pin_memory setting)
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Pin memory only helps with GPU transfer, not needed for CPU features
    pin_memory = not use_zfrac
    
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_tomato_loaders(data_root, batch_size=32, use_zfrac=False, 
                       grid_sizes=DEFAULT_GRID_SIZES, num_workers=0):
    """
    Create data loaders for the Tomato Leaf Disease dataset.
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size for data loaders
        use_zfrac: If True, use fractal features instead of raw images
        grid_sizes: List of grid sizes for fractal feature extraction
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, input_dim)
    """
    # Only use image transforms if not using fractal features
    train_transform = get_transforms(is_training=True) if not use_zfrac else None
    eval_transform = get_transforms(is_training=False) if not use_zfrac else None
    
    train_dataset = TomatoDataset(data_root, 'train', train_transform, use_zfrac, grid_sizes)
    val_dataset = TomatoDataset(data_root, 'val', eval_transform, use_zfrac, grid_sizes)
    test_dataset = TomatoDataset(data_root, 'test', eval_transform, use_zfrac, grid_sizes)
    
    # Normalize features using training set statistics
    if use_zfrac:
        mean, std = compute_normalization_statistics(train_dataset)
        train_dataset.set_normalization(mean, std)
        val_dataset.set_normalization(mean, std)
        test_dataset.set_normalization(mean, std)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, use_zfrac, num_workers
    )
    
    input_dim = get_feature_dim(grid_sizes) if use_zfrac else None
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes, input_dim


def get_kolektor_loaders(data_root, batch_size=32, use_zfrac=False, 
                         grid_sizes=DEFAULT_GRID_SIZES, num_workers=0):
    """
    Create data loaders for the Kolektor Surface Defect Dataset.
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size for data loaders
        use_zfrac: If True, use fractal features instead of raw images
        grid_sizes: List of grid sizes for fractal feature extraction
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, input_dim)
    """
    train_transform = get_transforms(is_training=True) if not use_zfrac else None
    eval_transform = get_transforms(is_training=False) if not use_zfrac else None
    
    train_dataset = KolektorSDD(data_root, 'train', train_transform, use_zfrac, grid_sizes)
    val_dataset = KolektorSDD(data_root, 'val', eval_transform, use_zfrac, grid_sizes)
    test_dataset = KolektorSDD(data_root, 'test', eval_transform, use_zfrac, grid_sizes)
    
    if use_zfrac:
        mean, std = compute_normalization_statistics(train_dataset)
        train_dataset.set_normalization(mean, std)
        val_dataset.set_normalization(mean, std)
        test_dataset.set_normalization(mean, std)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, use_zfrac, num_workers
    )
    
    input_dim = get_feature_dim(grid_sizes) if use_zfrac else None
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes, input_dim


def get_magnetic_loaders(data_root, batch_size=32, use_zfrac=False, 
                         grid_sizes=DEFAULT_GRID_SIZES, num_workers=0):
    """
    Create data loaders for the Magnetic Tile Defect dataset.
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size for data loaders
        use_zfrac: If True, use fractal features instead of raw images
        grid_sizes: List of grid sizes for fractal feature extraction
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, input_dim)
    """
    train_transform = get_transforms(is_training=True) if not use_zfrac else None
    eval_transform = get_transforms(is_training=False) if not use_zfrac else None
    
    train_dataset = MagneticTileDataset(data_root, 'train', train_transform, use_zfrac, grid_sizes)
    val_dataset = MagneticTileDataset(data_root, 'val', eval_transform, use_zfrac, grid_sizes)
    test_dataset = MagneticTileDataset(data_root, 'test', eval_transform, use_zfrac, grid_sizes)
    
    if use_zfrac:
        mean, std = compute_normalization_statistics(train_dataset)
        train_dataset.set_normalization(mean, std)
        val_dataset.set_normalization(mean, std)
        test_dataset.set_normalization(mean, std)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, use_zfrac, num_workers
    )
    
    input_dim = get_feature_dim(grid_sizes) if use_zfrac else None
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes, input_dim


def get_surface_cracks_loaders(data_root, batch_size=32, use_zfrac=False, 
                                grid_sizes=DEFAULT_GRID_SIZES, max_samples=DEFAULT_MAX_SAMPLES, 
                                num_workers=0):
    """
    Create data loaders for the Surface Cracks dataset.
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size for data loaders
        use_zfrac: If True, use fractal features instead of raw images
        grid_sizes: List of grid sizes for fractal feature extraction
        max_samples: Maximum number of samples to use (for memory management)
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, input_dim)
    """
    train_transform = get_transforms(is_training=True) if not use_zfrac else None
    eval_transform = get_transforms(is_training=False) if not use_zfrac else None
    
    train_dataset = SurfaceCracksDataset(
        data_root, 'train', train_transform, use_zfrac, grid_sizes, max_samples=max_samples
    )
    val_dataset = SurfaceCracksDataset(
        data_root, 'val', eval_transform, use_zfrac, grid_sizes, max_samples=max_samples
    )
    test_dataset = SurfaceCracksDataset(
        data_root, 'test', eval_transform, use_zfrac, grid_sizes, max_samples=max_samples
    )
    
    if use_zfrac:
        mean, std = compute_normalization_statistics(train_dataset)
        train_dataset.set_normalization(mean, std)
        val_dataset.set_normalization(mean, std)
        test_dataset.set_normalization(mean, std)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, use_zfrac, num_workers
    )
    
    input_dim = get_feature_dim(grid_sizes) if use_zfrac else None
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes, input_dim


def get_neu_metal_loaders(data_root, batch_size=32, use_zfrac=False, 
                          grid_sizes=DEFAULT_GRID_SIZES, num_workers=0):
    """
    Create data loaders for the NEU Metal Surface Defects dataset.
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size for data loaders
        use_zfrac: If True, use fractal features instead of raw images
        grid_sizes: List of grid sizes for fractal feature extraction
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, input_dim)
    """
    train_transform = get_transforms(is_training=True) if not use_zfrac else None
    eval_transform = get_transforms(is_training=False) if not use_zfrac else None
    
    train_dataset = NEUMetalSurfaceDataset(data_root, 'train', train_transform, use_zfrac, grid_sizes)
    val_dataset = NEUMetalSurfaceDataset(data_root, 'val', eval_transform, use_zfrac, grid_sizes)
    test_dataset = NEUMetalSurfaceDataset(data_root, 'test', eval_transform, use_zfrac, grid_sizes)
    
    if use_zfrac:
        mean, std = compute_normalization_statistics(train_dataset)
        train_dataset.set_normalization(mean, std)
        val_dataset.set_normalization(mean, std)
        test_dataset.set_normalization(mean, std)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, use_zfrac, num_workers
    )
    
    input_dim = get_feature_dim(grid_sizes) if use_zfrac else None
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes, input_dim