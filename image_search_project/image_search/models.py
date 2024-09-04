from .utils import extract_features

class Product(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    image = models.ImageField(upload_to='product_images/')
    feature_vector = models.JSONField(null=True, blank=True)

    def save(self, *args, **kwargs):
        # Save the image first to generate the file path
        super().save(*args, **kwargs)
        
        # Extract features from the image
        features = extract_features(self.image.path)
        self.feature_vector = features.tolist()  # Convert numpy array to list
        super().save(*args, **kwargs)  # Save again to store the feature vector
