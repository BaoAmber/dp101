import torch
import matplotlib.pyplot as plt
from data_loader import get_image_path


def evaluate_image(model, dataset, index, device='cpu'):
    """
    Evaluate a single image
    
    Args:
        model: Trained model
        dataset: Dataset object
        index: Image index
        device: Computing device
    
    Returns:
        predicted_label: Predicted label
        true_label: True label
        confidence: Prediction confidence
        file_path: File path
        file_name: File name
    """
    # Check if index is valid
    if index < 0 or index >= len(dataset):
        print(f"Error: Index {index} is out of range. Dataset size is {len(dataset)}")
        return None, None, None, None, None
    
    # Get image and label
    image, true_label = dataset[index]
    
    # Process label: convert to int if tensor, otherwise use directly
    if isinstance(true_label, torch.Tensor):
        true_label_value = true_label.item()
    else:
        true_label_value = int(true_label)
    
    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = predicted.item()
        confidence = confidence.item()
    
    # Get file path information
    file_path, file_name, images_path, labels_path = get_image_path(index, dataset)
    
    # Display results
    print("\n" + "=" * 50)
    print(f"Image Index: {index}")
    print(f"True Label: {true_label_value}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Prediction Confidence: {confidence * 100:.2f}%")
    print(f"File Name: {file_name}")
    print(f"Data File Path: {images_path}")
    print(f"Label File Path: {labels_path}")
    print("=" * 50)
    
    # Display image (need to denormalize)
    image_display = image.squeeze().cpu().numpy()
    # Denormalize: mean=0.1307, std=0.3081
    image_display = image_display * 0.3081 + 0.1307
    image_display = image_display.clip(0, 1)  # Ensure values are in [0, 1] range
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image_display, cmap='gray')
    plt.title(f'Index: {index} | True: {true_label_value} | Predicted: {predicted_label} | Confidence: {confidence*100:.2f}%')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return predicted_label, true_label_value, confidence, file_path, file_name

