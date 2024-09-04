from django.db import models

class Product(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    image = models.ImageField(upload_to='product_images/')
    feature_vector = models.JSONField(null=True, blank=True)  # To store the image feature vector

    def __str__(self):
        return self.title
