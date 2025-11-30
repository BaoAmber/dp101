import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNet
from data_loader import get_data_loaders


def train_model(epochs, device='cpu'):
    """
    Train the model
    
    Args:
        epochs: Number of training epochs
        device: Computing device ('cpu' or 'cuda')
    
    Returns:
        model: Trained model
        train_loader: Training data loader
        test_loader: Test data loader
        test_dataset: Test dataset
    """
    # Get data loaders
    train_loader, test_loader, test_dataset = get_data_loaders(batch_size=64)
    
    # Create model
    model = SimpleNet().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nStarting training with {epochs} epochs...")
    print("-" * 50)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Calculate average loss and accuracy
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Loss: {avg_loss:.4f} - '
              f'Accuracy: {accuracy:.2f}%')
    
    print("-" * 50)
    print("Training completed!\n")
    
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    print(f"Test set accuracy: {test_accuracy:.2f}%\n")
    
    return model, train_loader, test_loader, test_dataset

