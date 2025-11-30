import torch
from torchvision import datasets, transforms
import os


def get_data_loaders(batch_size=64):
    """
    Load MNIST dataset
    
    Args:
        batch_size: Batch size
    
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
        test_dataset: Test dataset (for getting file paths)
    """
    # Data preprocessing: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and standard deviation
    ])
    
    # Download and load training set
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test set
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader, test_dataset


def get_image_path(index, dataset):
    """
    Get file path information for the specified image index
    
    Args:
        index: Image index
        dataset: Dataset object
    
    Returns:
        file_path: File path
        file_name: File name
        images_path: Images file path
        labels_path: Labels file path
    """
    # MNIST dataset is stored in data/MNIST/raw/ directory
    data_dir = './data/MNIST/raw'
    
    # Determine file name prefix based on whether it's training or test set
    if dataset.train:
        prefix = 'train'
    else:
        prefix = 't10k'
    
    # MNIST raw file naming format
    # Training set: train-images-idx3-ubyte, train-labels-idx1-ubyte
    # Test set: t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
    images_file = f'{prefix}-images-idx3-ubyte'
    labels_file = f'{prefix}-labels-idx1-ubyte'
    
    images_path = os.path.join(data_dir, images_file)
    labels_path = os.path.join(data_dir, labels_file)
    
    # Build display file name
    file_name = f'{prefix}_image_{index}.png'
    file_path = os.path.join(data_dir, file_name)
    
    return file_path, file_name, images_path, labels_path

