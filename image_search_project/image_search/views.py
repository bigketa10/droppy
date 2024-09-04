import os
from django.conf import settings
from django.shortcuts import render
from .models import Product
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The file at {img_path} does not exist.")
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = model.predict(img_array)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

def search_similar_products(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg')
        
        # Save the uploaded image
        with open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        try:
            # Extract features from the uploaded image
            uploaded_image_features = extract_features(temp_image_path)

            # Compare with stored features and calculate similarity
            products = Product.objects.all()
            similarities = []
            for product in products:
                stored_features = np.array(product.feature_vector)
                similarity = cosine_similarity(
                    [uploaded_image_features], [stored_features]
                )[0][0]
                similarities.append((product, similarity))

            # Sort products by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_products = [product for product, _ in similarities[:10]]  # Top 10 results

            return render(request, 'image_search/results.html', {'products': similar_products})
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    
    return render(request, 'image_search/search.html')
