import numpy as np
import torch
from tqdm import tqdm
from sklearn.cross_decomposition import CCA


# Constants for CCA/CKA analysis
DEFAULT_N_COMPONENTS = 10
DEFAULT_VARIANCE_THRESHOLD = 0.99
DEFAULT_MAX_SAMPLES_FOR_ANALYSIS = 500
DEFAULT_MAX_FEATURES_FOR_ANALYSIS = 1000
DEFAULT_CCA_MAX_ITER = 500
EPSILON_FOR_NUMERICAL_STABILITY = 1e-10
LOW_CORRELATION_THRESHOLD_CKA = 0.3
LOW_CORRELATION_THRESHOLD_SVCCA = 0.5


def center_columns(feature_matrix):
    """
    Center the columns of a matrix (subtract column means).
    
    This is a preprocessing step for CCA and CKA analysis to remove mean effects.
    
    Args:
        feature_matrix: 2D numpy array where each column is a feature
    
    Returns:
        Centered matrix with zero-mean columns
    """
    column_means = feature_matrix.mean(axis=0)
    return feature_matrix - column_means


def cka_linear(feature_matrix_x, feature_matrix_y):
    """
    Compute linear Centered Kernel Alignment (CKA) between two feature representations.
    
    CKA measures the similarity between two representations by comparing how
    similar the similarity structures are. A value of 1 means identical representations,
    while 0 means completely independent.
    
    Args:
        feature_matrix_x: First feature matrix (samples x features)
        feature_matrix_y: Second feature matrix (samples x features)
    
    Returns:
        CKA similarity score between 0 and 1
    """
    # Center both matrices
    centered_x = center_columns(feature_matrix_x)
    centered_y = center_columns(feature_matrix_y)
    
    # Compute Gram matrices (similarity matrices)
    gram_matrix_x = centered_x @ centered_x.T
    gram_matrix_y = centered_y @ centered_y.T
    
    # Compute HSIC (Hilbert-Schmidt Independence Criterion) terms
    # HSIC_XY measures the similarity between the two representations
    hsic_xy = np.sum(gram_matrix_x * gram_matrix_y)
    
    # HSIC_XX and HSIC_YY are normalization terms
    hsic_xx = np.sum(gram_matrix_x ** 2)
    hsic_yy = np.sum(gram_matrix_y ** 2)
    
    # CKA is the normalized HSIC
    denominator = np.sqrt(hsic_xx * hsic_yy + EPSILON_FOR_NUMERICAL_STABILITY)
    cka_score = hsic_xy / denominator
    
    return cka_score


def svcca(feature_matrix_x, feature_matrix_y, n_components=DEFAULT_N_COMPONENTS, 
          var_threshold=DEFAULT_VARIANCE_THRESHOLD):
    """
    Compute Singular Vector Canonical Correlation Analysis (SVCCA) between two representations.
    
    SVCCA first reduces the dimensionality of both representations using SVD, keeping
    enough dimensions to capture a specified fraction of variance. Then it computes
    CCA on the reduced representations and returns the mean correlation.
    
    Args:
        feature_matrix_x: First feature matrix (samples x features)
        feature_matrix_y: Second feature matrix (samples x features)
        n_components: Maximum number of CCA components to compute
        var_threshold: Fraction of variance to retain in SVD reduction
    
    Returns:
        Mean CCA correlation between the representations
    """
    # Center both matrices
    centered_x = center_columns(feature_matrix_x)
    centered_y = center_columns(feature_matrix_y)
    
    # Perform SVD to get principal components
    left_singular_x, singular_values_x, _ = np.linalg.svd(centered_x, full_matrices=False)
    left_singular_y, singular_values_y, _ = np.linalg.svd(centered_y, full_matrices=False)
    
    # Calculate cumulative variance explained
    total_variance_x = np.sum(singular_values_x ** 2)
    total_variance_y = np.sum(singular_values_y ** 2)
    cumulative_variance_x = np.cumsum(singular_values_x ** 2) / total_variance_x
    cumulative_variance_y = np.cumsum(singular_values_y ** 2) / total_variance_y
    
    # Find number of components needed to capture var_threshold of variance
    num_components_x = np.searchsorted(cumulative_variance_x, var_threshold) + 1
    num_components_y = np.searchsorted(cumulative_variance_y, var_threshold) + 1
    
    # Reduce dimensionality using principal components
    reduced_x = left_singular_x[:, :num_components_x]
    reduced_y = left_singular_y[:, :num_components_y]
    
    # Determine number of CCA components to compute
    # Limited by: requested n_components, available components, and sample size
    num_samples = feature_matrix_x.shape[0]
    max_cca_components = min(n_components, num_components_x, num_components_y, num_samples // 2)
    
    if max_cca_components < 1:
        return 0.0
    
    # Perform CCA
    cca = CCA(n_components=max_cca_components, max_iter=DEFAULT_CCA_MAX_ITER)
    try:
        transformed_x, transformed_y = cca.fit_transform(reduced_x, reduced_y)
        
        # Compute correlation for each CCA component
        correlations = []
        for component_index in range(max_cca_components):
            correlation_matrix = np.corrcoef(
                transformed_x[:, component_index], 
                transformed_y[:, component_index]
            )
            correlation = correlation_matrix[0, 1]
            
            # Only include valid (non-NaN) correlations
            if not np.isnan(correlation):
                correlations.append(abs(correlation))
        
        # Return mean correlation across all components
        if correlations:
            return np.mean(correlations)
        else:
            return 0.0
    except Exception:
        # If CCA fails (e.g., numerical issues), return 0
        return 0.0


def extract_cnn_features(model, data_loader, layer_index, device):
    """
    Extract intermediate layer features from a CNN model.
    
    Args:
        model: CNN model with get_layer_features method
        data_loader: DataLoader providing images
        layer_index: Index of the layer to extract features from
        device: Device to run inference on
    
    Returns:
        Numpy array of extracted features (samples x features)
    """
    model.eval()
    feature_list = []
    
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc=f"extracting layer {layer_index}", leave=False):
            images = images.to(device)
            layer_features = model.get_layer_features(images, layer_index)
            # Flatten spatial dimensions, keep batch dimension
            flattened_features = layer_features.view(layer_features.size(0), -1)
            feature_list.append(flattened_features.cpu().numpy())
    
    # Stack all batches into a single array
    return np.vstack(feature_list)


def extract_zfrac_features(data_loader):
    """
    Extract fractal features from a DataLoader.
    
    Args:
        data_loader: DataLoader providing fractal features (not images)
    
    Returns:
        Numpy array of fractal features (samples x features)
    """
    feature_list = []
    for features, _ in data_loader:
        feature_list.append(features.numpy())
    return np.vstack(feature_list)


def run_cca_cka_analysis(cnn_model, cnn_loader, zfrac_loader, device, num_layers=4):
    """
    Run CCA/CKA analysis comparing CNN layer features with fractal features.
    
    This analysis helps understand whether CNNs learn representations similar to
    fractal features, providing insight into what patterns CNNs capture.
    
    Args:
        cnn_model: Trained CNN model to extract features from
        cnn_loader: DataLoader for CNN (provides images)
        zfrac_loader: DataLoader for fractal features
        device: Device to run analysis on
        num_layers: Number of CNN layers to analyze
    
    Returns:
        Dictionary with 'cka' and 'cca' lists containing similarity scores for each layer
    """
    print("\n" + "="*50)
    print("SVCCA/CKA Analysis: CNN layers vs ZFrac features")
    print("="*50)
    
    # Extract fractal features
    zfrac_features = extract_zfrac_features(zfrac_loader)
    print(f"ZFrac features shape: {zfrac_features.shape}")
    
    results = {'cka': [], 'cca': []}
    
    # Analyze each CNN layer
    for layer_index in range(num_layers):
        print(f"\nLayer {layer_index + 1}:")
        
        # Extract CNN features from this layer
        cnn_features = extract_cnn_features(cnn_model, cnn_loader, layer_index, device)
        print(f"  CNN features shape: {cnn_features.shape}")
        
        # Subsample to manageable size for analysis
        # Using fewer samples speeds up computation while maintaining statistical validity
        num_samples_to_use = min(
            len(zfrac_features), 
            len(cnn_features), 
            DEFAULT_MAX_SAMPLES_FOR_ANALYSIS
        )
        sample_indices = np.random.choice(
            len(zfrac_features), num_samples_to_use, replace=False
        )
        
        zfrac_subset = zfrac_features[sample_indices]
        cnn_subset = cnn_features[sample_indices]
        
        # If CNN features are too high-dimensional, randomly subsample features
        # This prevents memory issues and speeds up computation
        if cnn_subset.shape[1] > DEFAULT_MAX_FEATURES_FOR_ANALYSIS:
            feature_indices = np.random.permutation(cnn_subset.shape[1])[:DEFAULT_MAX_FEATURES_FOR_ANALYSIS]
            cnn_subset = cnn_subset[:, feature_indices]
        
        # Compute similarity metrics
        cka_score = cka_linear(zfrac_subset, cnn_subset)
        svcca_score = svcca(zfrac_subset, cnn_subset)
        
        results['cka'].append(cka_score)
        results['cca'].append(svcca_score)
        
        print(f"  CKA similarity: {cka_score:.4f}")
        print(f"  SVCCA correlation: {svcca_score:.4f}")
    
    # Print summary
    print("\n" + "-"*50)
    print("Summary:")
    mean_cka = np.mean(results['cka'])
    mean_svcca = np.mean(results['cca'])
    print(f"  Mean CKA across layers: {mean_cka:.4f}")
    print(f"  Mean SVCCA across layers: {mean_svcca:.4f}")
    print("-"*50)
    
    # Provide interpretation
    if mean_cka < LOW_CORRELATION_THRESHOLD_CKA and mean_svcca < LOW_CORRELATION_THRESHOLD_SVCCA:
        print("Conclusion: Low correlation suggests CNN does NOT encode fractal features")
    else:
        print("Conclusion: Some correlation between CNN and fractal features")
    
    return results

