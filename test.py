import os
from tensorflow.keras.models import load_model
from dataloader import prepare_data

# Load data
image_path = './dataset'
_, _, _, _, test_images, test_masks = prepare_data(image_path)

# Load trained model
model = load_model('./model/unet_model.h5', compile=False)

# Evaluate model
results = model.evaluate(test_images, test_masks, batch_size=1)
print("Test results:", results)
