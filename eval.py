import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, data_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in data_loader:
            images = images.permute(0, 3, 1, 2).to(device)  # Permute to (batch_size, channels, height, width)
            labels = labels.unsqueeze(1).to(device)  # Ensure labels have shape (batch_size, 1, height, width)

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            total_loss += loss.item()

            # Calculate accuracy (assuming binary classification)
            predicted = outputs > 0.5  # Thresholding for binary classification
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy
