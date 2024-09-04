from django.shortcuts import render
from .models import Product
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search_similar_products(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Save the uploaded image temporarily
        uploaded_image = request.FILES['image']
        img_path = 'temp/temp_image.jpg'
        with open(img_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Extract features from the uploaded image
        uploaded_image_features = extract_features(img_path)

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
    
    return render(request, 'image_search/search.html')
