import os
import json
import torch
import numpy as np

from models import ZFracNN, CNN
from datasets import get_tomato_loaders, get_kolektor_loaders, get_magnetic_loaders
from train import train_model, evaluate
from cca_cka_analysis import run_cca_cka_analysis
from fractal_features import get_feature_dim


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
    }
}

RESULTS_DIR = "results"
MODELS_DIR = "models"
BATCH_SIZE = 32
EPOCHS = 200
LR = 0.001
PATIENCE = 10
GRID_SIZES = [1, 2, 4]


def get_model_path(dataset_key, model_type):
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"{dataset_key}_{model_type}.pth")


def run_zfrac_experiment(dataset_key):
    cfg = DATASETS[dataset_key]
    print("\n" + "="*60)
    print(f"ZFRAC + NN - {cfg['name']}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader, num_classes, input_dim = cfg['loader'](
        cfg['path'], BATCH_SIZE, use_zfrac=True, grid_sizes=GRID_SIZES
    )
    
    print(f"num classes: {num_classes}, input dim: {input_dim}")
    
    model = ZFracNN(input_dim, num_classes, hidden=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params:,}")
    
    model_path = get_model_path(dataset_key, 'zfrac')
    
    if os.path.exists(model_path):
        print(f"loading cached model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        train_time = checkpoint['train_time']
    else:
        model, history = train_model(model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE)
        train_time = history['train_time']
        torch.save({
            'model_state': model.state_dict(),
            'train_time': train_time,
            'num_classes': num_classes,
            'input_dim': input_dim
        }, model_path)
        print(f"saved model to {model_path}")
    
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest accuracy: {test_acc:.2f}%")
    print(f"Training time: {train_time:.1f}s")
    
    return {
        'test_acc': test_acc,
        'train_time': train_time,
        'params': total_params,
        'model': model
    }


def run_cnn_experiment(dataset_key, backbone='resnet18'):
    cfg = DATASETS[dataset_key]
    print("\n" + "="*60)
    print(f"CNN ({backbone}) - {cfg['name']}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader, num_classes, _ = cfg['loader'](
        cfg['path'], BATCH_SIZE, use_zfrac=False
    )
    
    print(f"num classes: {num_classes}")
    
    model = CNN(num_classes, backbone=backbone, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params:,}")
    
    model_path = get_model_path(dataset_key, f'cnn_{backbone}')
    
    if os.path.exists(model_path):
        print(f"loading cached model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        train_time = checkpoint['train_time']
    else:
        model, history = train_model(model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE)
        train_time = history['train_time']
        torch.save({
            'model_state': model.state_dict(),
            'train_time': train_time,
            'num_classes': num_classes,
            'backbone': backbone
        }, model_path)
        print(f"saved model to {model_path}")
    
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest accuracy: {test_acc:.2f}%")
    print(f"Training time: {train_time:.1f}s")
    
    return {
        'test_acc': test_acc,
        'train_time': train_time,
        'params': total_params,
        'model': model
    }


def run_cca_cka_experiment(dataset_key, cnn_model):
    cfg = DATASETS[dataset_key]
    print("\n" + "-"*40)
    print(f"SVCCA/CKA Analysis - {cfg['name']}")
    print("-"*40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, _, cnn_loader, _, _ = cfg['loader'](cfg['path'], BATCH_SIZE, use_zfrac=False)
    _, _, zfrac_loader, _, _ = cfg['loader'](cfg['path'], BATCH_SIZE, use_zfrac=True, grid_sizes=GRID_SIZES)
    
    results = run_cca_cka_analysis(cnn_model, cnn_loader, zfrac_loader, device)
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("FRACTAL FEATURES VS CNN - MULTI-DATASET EXPERIMENT")
    print("="*60)
    print(f"\nSettings: epochs={EPOCHS}, patience={PATIENCE}, lr={LR}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    all_results = {}
    
    for dataset_key in DATASETS:
        print("\n" + "#"*60)
        print(f"# DATASET: {DATASETS[dataset_key]['name'].upper()}")
        print("#"*60)
        
        zfrac_results = run_zfrac_experiment(dataset_key)
        cnn_results = run_cnn_experiment(dataset_key)
        cca_cka = run_cca_cka_experiment(dataset_key, cnn_results['model'])
        
        all_results[dataset_key] = {
            'name': DATASETS[dataset_key]['name'],
            'zfrac': {k: v for k, v in zfrac_results.items() if k != 'model'},
            'cnn': {k: v for k, v in cnn_results.items() if k != 'model'},
            'cka_mean': float(np.mean(cca_cka['cka'])),
            'svcca_mean': float(np.mean(cca_cka['cca']))
        }
    
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL DATASETS")
    print("="*80)
    print(f"\n{'Dataset':<25} {'ZFrac Acc':<12} {'CNN Acc':<12} {'ZFrac Time':<12} {'CNN Time':<12} {'Speedup':<10}")
    print("-"*80)
    
    for key, r in all_results.items():
        speedup = r['cnn']['train_time'] / r['zfrac']['train_time']
        print(f"{r['name']:<25} {r['zfrac']['test_acc']:<12.2f} {r['cnn']['test_acc']:<12.2f} "
              f"{r['zfrac']['train_time']:<12.1f} {r['cnn']['train_time']:<12.1f} {speedup:<10.1f}x")
    
    print("\n" + "-"*80)
    print("SVCCA/CKA Analysis (CNN vs ZFrac features):")
    print("-"*80)
    for key, r in all_results.items():
        print(f"{r['name']:<25} CKA: {r['cka_mean']:.4f}  SVCCA: {r['svcca_mean']:.4f}")
    
    summary = {
        'settings': {'epochs': EPOCHS, 'patience': PATIENCE, 'lr': LR, 'batch_size': BATCH_SIZE},
        'results': all_results
    }
    
    with open(os.path.join(RESULTS_DIR, 'all_datasets_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}/all_datasets_results.json")


if __name__ == "__main__":
    main()
