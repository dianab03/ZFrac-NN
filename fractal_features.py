import numpy as np
from PIL import Image
import cv2


def box_counting_fd(gray_img):
    h, w = gray_img.shape
    size = min(h, w)
    
    if size < 16:
        return 2.0
    
    size = 2 ** int(np.log2(size))
    img = cv2.resize(gray_img, (size, size)).astype(np.float64)
    img = (img / 255.0) * (size - 1)
    
    box_sizes = []
    s = 2
    while s <= size // 2:
        box_sizes.append(s)
        s *= 2
    
    if len(box_sizes) < 2:
        return 2.0
    
    counts = []
    for s in box_sizes:
        nr = 0
        for i in range(0, size, s):
            for j in range(0, size, s):
                block = img[i:i+s, j:j+s]
                if block.size == 0:
                    continue
                min_val = block.min()
                max_val = block.max()
                k_min = int(min_val / s)
                k_max = int(max_val / s)
                nr += max(1, k_max - k_min + 1)
        counts.append(nr)
    
    box_sizes = np.array(box_sizes, dtype=np.float64)
    counts = np.array(counts, dtype=np.float64)
    
    valid = counts > 0
    if np.sum(valid) < 2:
        return 2.0
    
    log_s = np.log(1.0 / box_sizes[valid])
    log_n = np.log(counts[valid])
    
    coeffs = np.polyfit(log_s, log_n, 1)
    fd = coeffs[0]
    
    return np.clip(fd, 1.0, 3.0)


def extract_zfrac(image, grid_sizes=[1, 2, 4]):
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image = cv2.resize(image, (256, 256))
    h, w = image.shape
    features = []
    
    for gs in grid_sizes:
        zone_h = h // gs
        zone_w = w // gs
        
        for i in range(gs):
            for j in range(gs):
                y1, y2 = i * zone_h, (i + 1) * zone_h
                x1, x2 = j * zone_w, (j + 1) * zone_w
                zone = image[y1:y2, x1:x2]
                fd = box_counting_fd(zone)
                features.append(fd)
    
    return np.array(features, dtype=np.float32)


def get_feature_dim(grid_sizes=[1, 2, 4]):
    return sum(g * g for g in grid_sizes)


def normalize_features(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
    
    X_norm = (X - mean) / std
    return X_norm, mean, std
