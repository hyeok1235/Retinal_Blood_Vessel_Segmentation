import os
from model import UNet
from dataloader import prepare_data
from eval import evaluate_model
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Load data
image_path = './dataset'
_, _, _, _, test_images, test_masks = prepare_data(image_path)

test_loader = DataLoader(list(zip(test_images, test_masks)), batch_size=1, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('./model/unet_model.pth'))
model.eval()  # Set the model to evaluation mode

# Evaluate the model
criterion = torch.nn.BCELoss()  # or any other loss function you're using
avg_loss, accuracy = evaluate_model(model, test_loader, criterion)

print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
