import numpy as np
from PIL import Image
import cv2


# Constants for fractal dimension calculation
MIN_IMAGE_SIZE = 16  # Minimum image size required for box counting
MIN_BOX_SIZES_COUNT = 2  # Need at least 2 box sizes for linear regression
MIN_FRACTAL_DIMENSION = 1.0  # Minimum valid fractal dimension
MAX_FRACTAL_DIMENSION = 3.0  # Maximum valid fractal dimension
DEFAULT_FRACTAL_DIMENSION = 2.0  # Default when calculation fails
INITIAL_BOX_SIZE = 2  # Starting box size for box counting algorithm
MAX_PIXEL_VALUE = 255.0  # Maximum pixel value for normalization
RESIZE_TARGET_SIZE = 256  # Target size for image resizing in feature extraction


def box_counting_fd(gray_image):
    """
    Calculate the fractal dimension of a grayscale image using the box counting method.
    
    The box counting method divides the image into boxes of different sizes and counts
    how many boxes are needed to cover the image structure. The fractal dimension is
    calculated from the relationship between box size and box count.
    
    Args:
        gray_image: A 2D numpy array representing a grayscale image
        
    Returns:
        A float representing the fractal dimension, clipped between 1.0 and 3.0
    """
    image_height, image_width = gray_image.shape
    min_dimension = min(image_height, image_width)
    
    # If image is too small, return default value
    # Box counting requires sufficient resolution to be meaningful
    if min_dimension < MIN_IMAGE_SIZE:
        return DEFAULT_FRACTAL_DIMENSION
    
    # Round down to nearest power of 2 for consistent box sizing
    # This ensures we can divide the image evenly into boxes
    power_of_two_size = 2 ** int(np.log2(min_dimension))
    
    # Resize image to power-of-two dimensions and normalize pixel values
    # Normalize to range [0, size-1] for proper box counting
    resized_image = cv2.resize(gray_image, (power_of_two_size, power_of_two_size)).astype(np.float64)
    normalized_image = (resized_image / MAX_PIXEL_VALUE) * (power_of_two_size - 1)
    
    # Generate box sizes: start with 2, double each time until half the image size
    # We use powers of 2 to ensure clean divisions
    box_sizes_list = []
    current_box_size = INITIAL_BOX_SIZE
    max_box_size = power_of_two_size // 2
    
    while current_box_size <= max_box_size:
        box_sizes_list.append(current_box_size)
        current_box_size = current_box_size * 2
    
    # Need at least 2 box sizes to fit a line (calculate slope)
    if len(box_sizes_list) < MIN_BOX_SIZES_COUNT:
        return DEFAULT_FRACTAL_DIMENSION
    
    # Count boxes needed for each box size
    box_counts = []
    for box_size in box_sizes_list:
        boxes_needed = 0
        
        # Divide image into grid of boxes
        for row_start in range(0, power_of_two_size, box_size):
            for col_start in range(0, power_of_two_size, box_size):
                row_end = row_start + box_size
                col_end = col_start + box_size
                image_block = normalized_image[row_start:row_end, col_start:col_end]
                
                # Skip empty blocks
                if image_block.size == 0:
                    continue
                
                # Calculate how many "height layers" this box spans
                # This accounts for the 3D nature of the grayscale image
                block_min_value = image_block.min()
                block_max_value = image_block.max()
                min_height_layer = int(block_min_value / box_size)
                max_height_layer = int(block_max_value / box_size)
                
                # At least 1 box is needed, but may need more if values span multiple layers
                layers_spanned = max_height_layer - min_height_layer + 1
                boxes_needed += max(1, layers_spanned)
        
        box_counts.append(boxes_needed)
    
    # Convert to numpy arrays for mathematical operations
    box_sizes_array = np.array(box_sizes_list, dtype=np.float64)
    box_counts_array = np.array(box_counts, dtype=np.float64)
    
    # Filter out invalid counts (shouldn't happen, but safety check)
    valid_indices = box_counts_array > 0
    if np.sum(valid_indices) < MIN_BOX_SIZES_COUNT:
        return DEFAULT_FRACTAL_DIMENSION
    
    # Calculate fractal dimension using linear regression
    # Fractal dimension = slope of log(box_count) vs log(1/box_size)
    # This relationship follows: N(s) ~ s^(-D) where D is the fractal dimension
    log_inverse_box_sizes = np.log(1.0 / box_sizes_array[valid_indices])
    log_box_counts = np.log(box_counts_array[valid_indices])
    
    # Fit a line: log(N) = -D * log(s) + constant
    # The slope (first coefficient) is the negative of the fractal dimension
    regression_coefficients = np.polyfit(log_inverse_box_sizes, log_box_counts, 1)
    fractal_dimension = regression_coefficients[0]
    
    # Clip to valid range (fractal dimensions are typically between 1 and 3)
    return np.clip(fractal_dimension, MIN_FRACTAL_DIMENSION, MAX_FRACTAL_DIMENSION)


def extract_zfrac(image, grid_sizes=[1, 2, 4]):
    """
    Extract zonal fractal features from an image.
    
    The image is divided into zones based on grid_sizes, and the fractal dimension
    is calculated for each zone. This creates a multi-scale representation of the
    image's texture complexity.
    
    Args:
        image: A PIL Image, numpy array, or OpenCV image
        grid_sizes: List of integers specifying how many zones to divide the image into
                   (e.g., [1, 2, 4] means 1x1, 2x2, and 4x4 grids)
    
    Returns:
        A numpy array of fractal dimensions, one for each zone across all grid sizes
    """
    # Convert image to grayscale numpy array
    # Handle different input types for flexibility
    if isinstance(image, Image.Image):
        image_array = np.array(image.convert('L'))
    elif len(image.shape) == 3:
        # Convert RGB/BGR to grayscale
        image_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_array = image
    
    # Resize to standard size for consistent feature extraction
    # This ensures all images are processed at the same resolution
    resized_image = cv2.resize(image_array, (RESIZE_TARGET_SIZE, RESIZE_TARGET_SIZE))
    image_height, image_width = resized_image.shape
    
    fractal_features = []
    
    # Extract features for each grid size
    # Larger grid sizes capture more localized texture information
    for grid_size in grid_sizes:
        # Calculate zone dimensions
        zone_height = image_height // grid_size
        zone_width = image_width // grid_size
        
        # Extract fractal dimension for each zone in the grid
        for grid_row in range(grid_size):
            for grid_col in range(grid_size):
                # Calculate zone boundaries
                zone_row_start = grid_row * zone_height
                zone_row_end = (grid_row + 1) * zone_height
                zone_col_start = grid_col * zone_width
                zone_col_end = (grid_col + 1) * zone_width
                
                # Extract the zone (sub-image)
                image_zone = resized_image[zone_row_start:zone_row_end, zone_col_start:zone_col_end]
                
                # Calculate fractal dimension for this zone
                zone_fractal_dimension = box_counting_fd(image_zone)
                fractal_features.append(zone_fractal_dimension)
    
    return np.array(fractal_features, dtype=np.float32)


def get_feature_dim(grid_sizes=[1, 2, 4]):
    """
    Calculate the total number of features that will be extracted.
    
    For each grid size, we create a grid_size x grid_size array of zones,
    so the total number of features is the sum of grid_size^2 for each grid size.
    
    Args:
        grid_sizes: List of integers specifying grid divisions
        
    Returns:
        Total number of fractal dimension features that will be extracted
    """
    total_features = 0
    for grid_size in grid_sizes:
        # Each grid size creates grid_size x grid_size zones
        zones_per_grid = grid_size * grid_size
        total_features += zones_per_grid
    
    return total_features


def normalize_features(feature_matrix, mean=None, std=None):
    """
    Normalize features using z-score normalization (zero mean, unit variance).
    
    This is important for machine learning models as it ensures all features
    are on the same scale, preventing features with larger values from
    dominating the learning process.
    
    Args:
        feature_matrix: 2D numpy array where each row is a sample and each column is a feature
        mean: Optional pre-computed mean (if None, will compute from feature_matrix)
        std: Optional pre-computed standard deviation (if None, will compute from feature_matrix)
    
    Returns:
        Tuple of (normalized_features, mean, std) for potential reuse on test data
    """
    # Compute mean and std from training data if not provided
    # This allows us to normalize test data using training statistics
    if mean is None:
        mean = np.mean(feature_matrix, axis=0)
    
    if std is None:
        std = np.std(feature_matrix, axis=0)
        # Prevent division by zero for constant features
        # Set std to 1.0 so normalization becomes (x - mean) / 1.0 = x - mean
        std[std == 0] = 1.0
    
    # Z-score normalization: (x - mean) / std
    normalized_features = (feature_matrix - mean) / std
    
    return normalized_features, mean, std
