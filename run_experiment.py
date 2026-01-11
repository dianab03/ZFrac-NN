import os
import json
import torch
import numpy as np

from models import ZFracNN, CNN
from datasets import get_tomato_loaders, get_kolektor_loaders, get_magnetic_loaders, get_surface_cracks_loaders, get_neu_metal_loaders
from train import train_model, evaluate
from cca_cka_analysis import run_cca_cka_analysis
from fractal_features import get_feature_dim


# Dataset configuration: maps dataset keys to their paths, loaders, and display names
DATASETS = {
    'tomato': {
        'path': 'dataset/Tomato Leaf Disease Dataset/TomatoDataset',
        'loader': get_tomato_loaders,
        'name': 'Tomato Leaf Disease'
    },
    'kolektor': {
        'path': 'dataset/KolektorSSD',
        'loader': get_kolektor_loaders,
        'name': 'KolektorSDD'
    },
    'magnetic': {
        'path': 'dataset/MagneticTileDefect',
        'loader': get_magnetic_loaders,
        'name': 'Magnetic Tile Defect'
    },
    'surface_cracks': {
        'path': 'dataset/surface_cracks',
        # Surface cracks needs max_samples parameter, so we wrap the loader
        'loader': lambda data_root, batch_size=32, use_zfrac=False, grid_sizes=[1, 2, 4], num_workers=0: 
                 get_surface_cracks_loaders(data_root, batch_size, use_zfrac, grid_sizes, max_samples=10000, num_workers=num_workers),
        'name': 'Surface Cracks'
    },
    'neu_metal': {
        'path': 'dataset/NEU_MetalSurfaceDefects',
        'loader': get_neu_metal_loaders,
        'name': 'NEU Metal Surface Defects'
    }
}

# Experiment configuration constants
RESULTS_DIR = "results"
MODELS_DIR = "models"
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
GRID_SIZES = [1, 2, 4]
CNN_BACKBONES = ['resnet18', 'vgg16', 'densenet121']


def get_model_path(dataset_key, model_type):
    """
    Generate the file path for saving/loading a trained model.
    
    Args:
        dataset_key: Key identifying the dataset (e.g., 'tomato', 'kolektor')
        model_type: Type of model (e.g., 'zfrac', 'cnn_resnet18')
    
    Returns:
        Full path to the model file
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_filename = f"{dataset_key}_{model_type}.pth"
    return os.path.join(MODELS_DIR, model_filename)


def run_zfrac_experiment(dataset_key):
    """
    Run experiment using fractal features with a neural network.
    
    This trains a simple feedforward network on pre-extracted fractal features
    and evaluates its performance on the test set.
    
    Args:
        dataset_key: Key identifying which dataset to use
    
    Returns:
        Dictionary containing test accuracy, training time, parameter count, and model
    """
    dataset_config = DATASETS[dataset_key]
    print("\n" + "="*60)
    print(f"ZFRAC + NN - {dataset_config['name']}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data using fractal features
    train_loader, val_loader, test_loader, num_classes, input_dim = dataset_config['loader'](
        dataset_config['path'], BATCH_SIZE, use_zfrac=True, grid_sizes=GRID_SIZES
    )
    
    print(f"Number of classes: {num_classes}, Input dimension: {input_dim}")
    
    # Create model
    model = ZFracNN(input_dim, num_classes, hidden=128)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total parameters: {total_parameters:,}")
    
    model_path = get_model_path(dataset_key, 'zfrac')
    
    # Try to load cached model, otherwise train
    if os.path.exists(model_path):
        print(f"Loading cached model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        training_time = checkpoint['train_time']
    else:
        model, training_history = train_model(
            model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device, EARLY_STOPPING_PATIENCE
        )
        training_time = training_history['train_time']
        
        # Save model for future use
        torch.save({
            'model_state': model.state_dict(),
            'train_time': training_time,
            'num_classes': num_classes,
            'input_dim': input_dim
        }, model_path)
        print(f"Saved model to {model_path}")
    
    # Evaluate on test set
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy, predictions, true_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest accuracy: {test_accuracy:.2f}%")
    print(f"Training time: {training_time:.1f}s")
    
    return {
        'test_acc': test_accuracy,
        'train_time': training_time,
        'params': total_parameters,
        'model': model
    }


def run_cnn_experiment(dataset_key, backbone='resnet18'):
    """
    Run experiment using a CNN with pretrained backbone.
    
    This trains a CNN (ResNet, VGG, or DenseNet) on raw images and evaluates
    its performance on the test set.
    
    Args:
        dataset_key: Key identifying which dataset to use
        backbone: CNN architecture to use ('resnet18', 'vgg16', or 'densenet121')
    
    Returns:
        Dictionary containing test accuracy, training time, parameter count, and model
    """
    dataset_config = DATASETS[dataset_key]
    print("\n" + "="*60)
    print(f"CNN ({backbone}) - {dataset_config['name']}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data using raw images (not fractal features)
    train_loader, val_loader, test_loader, num_classes, _ = dataset_config['loader'](
        dataset_config['path'], BATCH_SIZE, use_zfrac=False
    )
    
    print(f"Number of classes: {num_classes}")
    
    # Create model with pretrained backbone
    model = CNN(num_classes, backbone=backbone, pretrained=True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total parameters: {total_parameters:,}")
    
    model_path = get_model_path(dataset_key, f'cnn_{backbone}')
    
    # Try to load cached model, otherwise train
    if os.path.exists(model_path):
        print(f"Loading cached model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        training_time = checkpoint['train_time']
    else:
        model, training_history = train_model(
            model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device, EARLY_STOPPING_PATIENCE
        )
        training_time = training_history['train_time']
        
        # Save model for future use
        torch.save({
            'model_state': model.state_dict(),
            'train_time': training_time,
            'num_classes': num_classes,
            'backbone': backbone
        }, model_path)
        print(f"Saved model to {model_path}")
    
    # Evaluate on test set
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy, predictions, true_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest accuracy: {test_accuracy:.2f}%")
    print(f"Training time: {training_time:.1f}s")
    
    return {
        'test_acc': test_accuracy,
        'train_time': training_time,
        'params': total_parameters,
        'model': model
    }


def run_cca_cka_experiment(dataset_key, cnn_model):
    """
    Run CCA/CKA analysis to compare CNN and fractal feature representations.
    
    This analyzes how similar the learned CNN features are to the fractal features,
    providing insight into whether CNNs implicitly learn fractal-like patterns.
    
    Args:
        dataset_key: Key identifying which dataset to use
        cnn_model: Trained CNN model to extract features from
    
    Returns:
        Dictionary containing CKA and SVCCA similarity scores
    """
    dataset_config = DATASETS[dataset_key]
    print("\n" + "-"*40)
    print(f"SVCCA/CKA Analysis - {dataset_config['name']}")
    print("-"*40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data for both CNN (images) and fractal features
    _, _, cnn_loader, _, _ = dataset_config['loader'](
        dataset_config['path'], BATCH_SIZE, use_zfrac=False
    )
    _, _, zfrac_loader, _, _ = dataset_config['loader'](
        dataset_config['path'], BATCH_SIZE, use_zfrac=True, grid_sizes=GRID_SIZES
    )
    
    analysis_results = run_cca_cka_analysis(cnn_model, cnn_loader, zfrac_loader, device)
    return analysis_results


def extract_results_without_model(results_dict):
    """
    Extract results dictionary excluding the model object.
    
    Models are large and shouldn't be saved in results JSON.
    
    Args:
        results_dict: Dictionary that may contain a 'model' key
    
    Returns:
        Dictionary with 'model' key removed
    """
    filtered_results = {}
    for key, value in results_dict.items():
        if key != 'model':
            filtered_results[key] = value
    return filtered_results


def print_results_table(all_results):
    """
    Print formatted results table comparing ZFrac and CNN performance.
    
    Args:
        all_results: Dictionary containing results for all datasets
    """
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL DATASETS")
    print("="*80)
    
    # Print results table for each CNN backbone
    for backbone in CNN_BACKBONES:
        print(f"\n{'Dataset':<25} {'ZFrac Acc':<12} {f'{backbone.upper()} Acc':<15} "
              f"{'ZFrac Time':<12} {f'{backbone.upper()} Time':<15} {'Speedup':<10}")
        print("-"*80)
        
        for dataset_key, dataset_results in all_results.items():
            if backbone in dataset_results['cnn']:
                zfrac_accuracy = dataset_results['zfrac']['test_acc']
                cnn_accuracy = dataset_results['cnn'][backbone]['test_acc']
                zfrac_time = dataset_results['zfrac']['train_time']
                cnn_time = dataset_results['cnn'][backbone]['train_time']
                speedup = cnn_time / zfrac_time
                
                print(f"{dataset_results['name']:<25} {zfrac_accuracy:<12.2f} {cnn_accuracy:<15.2f} "
                      f"{zfrac_time:<12.1f} {cnn_time:<15.1f} {speedup:<10.1f}x")
    
    # Print CCA/CKA analysis results
    print("\n" + "-"*80)
    print("SVCCA/CKA Analysis (ResNet18 vs ZFrac features):")
    print("-"*80)
    for dataset_key, dataset_results in all_results.items():
        dataset_name = dataset_results['name']
        cka_mean = dataset_results['cka_mean']
        svcca_mean = dataset_results['svcca_mean']
        print(f"{dataset_name:<25} CKA: {cka_mean:.4f}  SVCCA: {svcca_mean:.4f}")


def main():
    """
    Main function to run experiments across all datasets.
    
    For each dataset, this function:
    1. Trains a ZFrac+NN model
    2. Trains CNN models with different backbones
    3. Performs CCA/CKA analysis comparing CNN and fractal features
    4. Prints and saves results
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("FRACTAL FEATURES VS CNN - MULTI-DATASET EXPERIMENT")
    print("="*60)
    print(f"\nSettings: epochs={EPOCHS}, patience={EARLY_STOPPING_PATIENCE}, lr={LEARNING_RATE}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    all_results = {}
    
    # Run experiments for each dataset
    for dataset_key in DATASETS:
        print("\n" + "#"*60)
        dataset_name = DATASETS[dataset_key]['name']
        print(f"# DATASET: {dataset_name.upper()}")
        print("#"*60)
        
        # Run ZFrac experiment
        zfrac_results = run_zfrac_experiment(dataset_key)
        
        # Test all CNN backbones
        cnn_results_by_backbone = {}
        for backbone in CNN_BACKBONES:
            cnn_results = run_cnn_experiment(dataset_key, backbone=backbone)
            # Store results without the model object
            cnn_results_by_backbone[backbone] = extract_results_without_model(cnn_results)
        
        # Use ResNet18 for CCA/CKA analysis (as baseline comparison)
        cnn_results_resnet = run_cnn_experiment(dataset_key, backbone='resnet18')
        cca_cka_results = run_cca_cka_experiment(dataset_key, cnn_results_resnet['model'])
        
        # Store all results for this dataset
        all_results[dataset_key] = {
            'name': dataset_name,
            'zfrac': extract_results_without_model(zfrac_results),
            'cnn': cnn_results_by_backbone,
            'cka_mean': float(np.mean(cca_cka_results['cka'])),
            'svcca_mean': float(np.mean(cca_cka_results['cca']))
        }
    
    # Print formatted results table
    print_results_table(all_results)
    
    # Save results to JSON file
    summary = {
        'settings': {
            'epochs': EPOCHS, 
            'patience': EARLY_STOPPING_PATIENCE, 
            'lr': LEARNING_RATE, 
            'batch_size': BATCH_SIZE
        },
        'results': all_results
    }
    
    results_file_path = os.path.join(RESULTS_DIR, 'all_datasets_results.json')
    with open(results_file_path, 'w') as results_file:
        json.dump(summary, results_file, indent=2)
    
    print(f"\nResults saved to {results_file_path}")


if __name__ == "__main__":
    main()
