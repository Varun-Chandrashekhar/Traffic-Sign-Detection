import numpy as np
from PIL import Image

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 32x32
    image = image.resize((32, 32))
    # Normalize
    image = np.array(image)
    image = image / 255.0
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    image = image.reshape(1, 32, 32, 1)
    return image