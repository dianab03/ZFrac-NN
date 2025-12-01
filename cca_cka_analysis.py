import numpy as np
import torch
from tqdm import tqdm
from sklearn.cross_decomposition import CCA


def center_columns(X):
    return X - X.mean(axis=0)


def cka_linear(X, Y):
    X = center_columns(X)
    Y = center_columns(Y)
    
    hsic_xy = np.sum((X @ X.T) * (Y @ Y.T))
    hsic_xx = np.sum((X @ X.T) ** 2)
    hsic_yy = np.sum((Y @ Y.T) ** 2)
    
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10)


def svcca(X, Y, n_components=10, var_threshold=0.99):
    X = center_columns(X)
    Y = center_columns(Y)
    
    U_x, s_x, _ = np.linalg.svd(X, full_matrices=False)
    U_y, s_y, _ = np.linalg.svd(Y, full_matrices=False)
    
    var_x = np.cumsum(s_x ** 2) / np.sum(s_x ** 2)
    var_y = np.cumsum(s_y ** 2) / np.sum(s_y ** 2)
    
    k_x = np.searchsorted(var_x, var_threshold) + 1
    k_y = np.searchsorted(var_y, var_threshold) + 1
    
    X_reduced = U_x[:, :k_x]
    Y_reduced = U_y[:, :k_y]
    
    n_comp = min(n_components, k_x, k_y, X.shape[0] // 2)
    
    if n_comp < 1:
        return 0.0
    
    cca = CCA(n_components=n_comp, max_iter=500)
    try:
        X_c, Y_c = cca.fit_transform(X_reduced, Y_reduced)
        corrs = []
        for i in range(n_comp):
            c = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            if not np.isnan(c):
                corrs.append(abs(c))
        return np.mean(corrs) if corrs else 0.0
    except:
        return 0.0


def extract_cnn_features(model, loader, layer_idx, device):
    model.eval()
    features = []
    
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"extracting layer {layer_idx}", leave=False):
            x = x.to(device)
            feat = model.get_layer_features(x, layer_idx)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())
    
    return np.vstack(features)


def extract_zfrac_features(loader):
    features = []
    for x, _ in loader:
        features.append(x.numpy())
    return np.vstack(features)


def run_cca_cka_analysis(cnn_model, cnn_loader, zfrac_loader, device, num_layers=4):
    print("\n" + "="*50)
    print("SVCCA/CKA Analysis: CNN layers vs ZFrac features")
    print("="*50)
    
    zfrac_feats = extract_zfrac_features(zfrac_loader)
    print(f"ZFrac features shape: {zfrac_feats.shape}")
    
    results = {'cka': [], 'cca': []}
    
    for layer_idx in range(num_layers):
        print(f"\nLayer {layer_idx + 1}:")
        
        cnn_feats = extract_cnn_features(cnn_model, cnn_loader, layer_idx, device)
        print(f"  CNN features shape: {cnn_feats.shape}")
        
        n_samples = min(len(zfrac_feats), len(cnn_feats), 500)
        idx = np.random.choice(len(zfrac_feats), n_samples, replace=False)
        
        z_sub = zfrac_feats[idx]
        c_sub = cnn_feats[idx]
        
        if c_sub.shape[1] > 1000:
            perm = np.random.permutation(c_sub.shape[1])[:1000]
            c_sub = c_sub[:, perm]
        
        cka_score = cka_linear(z_sub, c_sub)
        cca_score = svcca(z_sub, c_sub)
        
        results['cka'].append(cka_score)
        results['cca'].append(cca_score)
        
        print(f"  CKA similarity: {cka_score:.4f}")
        print(f"  SVCCA correlation: {cca_score:.4f}")
    
    print("\n" + "-"*50)
    print("Summary:")
    print(f"  Mean CKA across layers: {np.mean(results['cka']):.4f}")
    print(f"  Mean SVCCA across layers: {np.mean(results['cca']):.4f}")
    print("-"*50)
    
    if np.mean(results['cka']) < 0.3 and np.mean(results['cca']) < 0.5:
        print("Conclusion: Low correlation suggests CNN does NOT encode fractal features")
    else:
        print("Conclusion: Some correlation between CNN and fractal features")
    
    return results

