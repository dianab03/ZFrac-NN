# Fractal Features vs CNNs

Replication of "On The Potential of The Fractal Geometry and The CNNs' Ability to Encode it"

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py
```

This will:
1. Train ZFrac + shallow NN
2. Train ResNet18 CNN
3. Run CCA/CKA analysis to compare CNN features vs fractal features
4. Print comparison results

## Files

- `fractal_features.py` - ZFrac extraction using box counting
- `models.py` - ZFracNN and CNN models
- `datasets.py` - dataset loader
- `train.py` - training functions
- `cca_cka_analysis.py` - CCA/CKA similarity analysis
- `run_experiment.py` - main script
