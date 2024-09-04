from django.db import models
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load a pre-trained model
model = VGG16(weights='imagenet', include_top=False)

class Product(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    image = models.ImageField(upload_to='product_images/')
    feature_vector = models.JSONField(null=True, blank=True)  # To store the image feature vector

    def save(self, *args, **kwargs):
        # Save the image first to generate the file path
        super().save(*args, **kwargs)
        
        # Extract features from the image
        features = extract_features(self.image.path)
        self.feature_vector = features.tolist()  # Convert numpy array to list
        super().save(*args, **kwargs)  # Save again to store the feature vector

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = model.predict(img_array)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features
