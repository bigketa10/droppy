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
def search_view(request):
    if request.method == 'POST':
        # Handle image upload
        uploaded_image = request.FILES['image']
        temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg')

        # Save the uploaded image to media directory
        with open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        print(f"Uploaded image saved at: {temp_image_path}")
        
        try:
            uploaded_image_features = extract_features(temp_image_path)
            print(f"Extracted features: {uploaded_image_features}")  # Debugging output
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            return render(request, 'image_search/search.html', {'error': 'Feature extraction failed.'})
        
        products = Product.objects.all()
        for product in products:
            print(f"Product: {product.name}, Feature Vector: {product.feature_vector}")  # Debugging output
        # Calculate similarity
        
        similarities = []
        for product in products:
            stored_features = np.array(product.feature_vector)
            similarity = cosine_similarity([uploaded_image_features], [stored_features])[0][0]
            print(f"Similarity with {product.name}: {similarity}")  # Debugging output
            similarities.append((product, similarity))

        # Filter based on similarity threshold
        threshold = 0.5  # Adjust for testing
        similar_products = [product for product, sim in similarities if sim > threshold]

        if not similar_products:
            print("No similar products found.")  # Debugging output
            return render(request, 'image_search/search.html', {'message': 'No similar products found'})


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
    temp_image_path = None
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg')
        
        # Save the uploaded image
        try:
            with open(temp_image_path, 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)
        except IOError as e:
            return render(request, 'image_search/search.html', {'error': f'File save error: {e}'})
        
        # Proceed if file saved successfully
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
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                print(f"Temporary file path: {temp_image_path}")
                print(f"File exists: {os.path.exists(temp_image_path)}")

    
    return render(request, 'image_search/search.html')
