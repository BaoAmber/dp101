import torch
from train import train_model
from evaluate import evaluate_image
from data_loader import get_data_loaders


def main():
    """Main program entry point"""
    print("=" * 60)
    print("MNIST Handwritten Digit Recognition System")
    print("=" * 60)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Get test dataset (for evaluation)
    _, _, test_dataset = get_data_loaders()
    
    # Training phase
    print("=" * 60)
    print("Training Phase")
    print("=" * 60)
    
    while True:
        try:
            epochs = int(input("Enter number of training epochs (enter 0 to skip training): "))
            if epochs < 0:
                print("Please enter a non-negative integer!")
                continue
            break
        except ValueError:
            print("Please enter a valid number!")
            continue
    
    model = None
    if epochs > 0:
        model, _, test_loader, test_dataset = train_model(epochs, device)
        
        # Save model
        model_path = 'mnist_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}\n")
    else:
        # Try to load saved model
        try:
            from model import SimpleNet
            model = SimpleNet().to(device)
            model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
            model.eval()
            print("Loaded saved model.\n")
        except FileNotFoundError:
            print("Error: Saved model not found. Please train the model first!")
            return
    
    # Evaluation phase
    print("=" * 60)
    print("Evaluation Mode")
    print("=" * 60)
    print(f"Test set size: {len(test_dataset)}")
    print("Enter image index (0-9999) to view prediction results")
    print("Enter 'q' or 'quit' to exit the program\n")
    
    while True:
        user_input = input("Enter image index: ").strip().lower()
        
        if user_input in ['q', 'quit', 'exit']:
            print("Program exited.")
            break
        
        try:
            index = int(user_input)
            evaluate_image(model, test_dataset, index, device)
            print()  # Empty line separator
        except ValueError:
            print("Please enter a valid number index or 'q' to exit!\n")
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}\n")


if __name__ == '__main__':
    main()

