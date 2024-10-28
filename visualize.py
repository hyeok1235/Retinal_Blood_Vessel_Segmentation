import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from model import UNet

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('./model/unet_model.pth'))

model.eval()  # Set the model to evaluation mode

# Define path to a test image
image_path = "./dataset/test/Original"
img_path = os.path.join(image_path, '20_A.png')
img = Image.open(img_path).convert("RGB")
img = img.resize((256, 256))
img = np.array(img, dtype=np.float32) / 255.0

# Expand dimensions to fit the model input
img = np.expand_dims(img, axis=0)  # Shape: (1, height, width, channels)

# Preprocess the image: Convert it to a tensor and permute dimensions
img_tensor = torch.from_numpy(img).permute(0, 3, 1, 2).to(device)  # Shape: (1, 3, height, width)

# Make prediction
with torch.no_grad():  # Disable gradient calculations
    predicted_mask = model(img_tensor)  # Forward pass

# Post-process the output if necessary (e.g., apply sigmoid for binary mask)
predicted_mask = torch.sigmoid(predicted_mask).cpu().numpy()  # Shape: (1, 1, height, width)
predicted_mask = np.squeeze(predicted_mask)  # Remove the channel dimension

# Visualize the predicted mask
plt.figure(figsize=(16, 8))
# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(img[0])  # Adjust `cmap` based on your image type
plt.axis('off')
plt.title('Original Image')

# Display the predicted mask
plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.axis('off')
plt.title('Predicted Mask')
plt.show()
