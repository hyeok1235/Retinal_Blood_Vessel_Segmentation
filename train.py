import os
import tensorflow as tf
from model import UNet
from dataloader import prepare_data
import torch

import torch.optim as optim
import torch.nn as nn

# Load data
image_path = "./dataset"
train_images, train_masks, images_val, masks_val, _, _ = prepare_data(image_path)

# Model setup
model = UNet()

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary cross-entropy for segmentation

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Convert data to PyTorch tensors and move to the same device as the model
train_images = torch.tensor(train_images, dtype=torch.float32).to(device)
train_masks = torch.tensor(train_masks, dtype=torch.float32).to(device)
images_val = torch.tensor(images_val, dtype=torch.float32).to(device)
masks_val = torch.tensor(masks_val, dtype=torch.float32).to(device)

# Set model to training mode
model.train()

# Training loop
epochs = 10
batch_size = 1

print("Training the model...")
for epoch in range(epochs):
    running_loss = 0.0
    for i in range(0, len(train_images), batch_size):
      # images : RGB (channel=3), masks : grayscale (channel=1)
      inputs = train_images[i:i+batch_size].to(device) # torch.Size([batch_size, 256, 256, 3])
      masks = train_masks[i:i+batch_size].to(device) # torch.Size([batch_size, 256, 256])
      masks = masks.unsqueeze(1)  # This makes masks have shape (batch_size, 1, 256, 256)

      # Permute the inputs to match PyTorch's expected input format
      inputs = inputs.permute(0, 3, 1, 2)

      # Zero the parameter gradients
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, masks)

      # Backward pass and optimize
      loss.backward()
      optimizer.step()

      running_loss += loss.item()


    # Print epoch loss
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_images)}")

print("Training complete.")

# Save the model
os.makedirs('./model', exist_ok=True)
model.save('./model/unet_model_epoch10.pth')
