import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model to train
        data_loader: DataLoader providing training batches
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: Device to run training on (CPU or GPU)
    
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, leave=False, desc="train")
    for images, labels in progress_bar:
        # Move data to device (GPU if available)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        predicted_classes = predictions.argmax(1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += batch_size
        
        # Update progress bar with current loss
        progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    average_loss = total_loss / total_samples
    accuracy_percentage = 100 * correct_predictions / total_samples
    
    return average_loss, accuracy_percentage


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The neural network model to evaluate
        data_loader: DataLoader providing evaluation batches
        criterion: Loss function
        device: Device to run evaluation on (CPU or GPU)
    
    Returns:
        Tuple of (average_loss, accuracy_percentage, predictions_array, true_labels_array)
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, leave=False, desc="eval"):
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            # Accumulate metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            predicted_classes = predictions.argmax(1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += batch_size
            
            # Store predictions and labels for later analysis
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    average_loss = total_loss / total_samples
    accuracy_percentage = 100 * correct_predictions / total_samples
    predictions_array = np.array(all_predictions)
    labels_array = np.array(all_true_labels)
    
    return average_loss, accuracy_percentage, predictions_array, labels_array


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, 
                device='cuda', patience=3):
    """
    Train a model with early stopping based on validation loss.
    
    The model is trained for a maximum number of epochs, but training stops early
    if validation loss doesn't improve for 'patience' consecutive epochs. The best
    model (based on validation loss) is restored at the end.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs to train
        learning_rate: Learning rate for the optimizer
        device: Device to run training on (CPU or GPU)
        patience: Number of epochs to wait for improvement before stopping
    
    Returns:
        Tuple of (trained_model, training_history_dict)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track best performance for early stopping
    best_validation_loss = float('inf')
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0
    best_model_state = None
    
    # Store training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on validation set
        val_loss, val_accuracy, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f"  Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2f}%")
        print(f"  Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2f}%")
        
        # Check if this is the best model so far
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_validation_accuracy = val_accuracy
            epochs_without_improvement = 0
            
            # Save the best model state
            # Move to CPU to save memory and allow loading on different devices
            best_model_state = {
                key: value.cpu().clone() 
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            
            # Early stopping: stop if no improvement for 'patience' epochs
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - training_start_time
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Add final metrics to history
    history['train_time'] = training_time
    history['best_val_acc'] = best_validation_accuracy
    history['best_val_loss'] = best_validation_loss
    
    return model, history
