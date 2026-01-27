import numpy as np
from scipy.spatial.distance import cosine
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16

# Preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Load model and features
def load_saved_model(model_path="vgg_features.pkl"):
    with open(model_path, "rb") as file:
        data = pickle.load(file)
    return data["features"], data["labels"]

# Compare features for similarity
def check_similarity(test_image_path, model, saved_features, saved_labels):
    img = preprocess_image(test_image_path)
    test_features = model.predict(img).flatten()

    similarities = [1 - cosine(test_features, train_feature) for train_feature in saved_features]
    best_match_idx = np.argmax(similarities)
    best_similarity_score = similarities[best_match_idx]

    print(f"Best Similarity Score: {best_similarity_score:.2f}")
    if best_similarity_score > 0.97:
        print(f"Image is similar to label: {saved_labels[best_match_idx]}")
    else:
        print("No similar image found.")

# Main workflow
if __name__ == "__main__":
    test_image_path = "1.jpg"  # Replace with your test image path

    # Load pre-trained VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Load saved features and labels
    saved_features, saved_labels = load_saved_model()

    # Check similarity
    check_similarity(test_image_path, vgg_model, saved_features, saved_labels)
