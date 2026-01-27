import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# Preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Extract features using VGG16
def extract_features(image_paths, model):
    features = []
    for path in image_paths:
        img = preprocess_image(path)
        feature = model.predict(img).flatten()
        features.append(feature)
    return np.array(features)

# Save model and features
def save_model_and_features(features, labels, output_path="vgg_features.pkl"):
    data = {"features": features, "labels": labels}
    with open(output_path, "wb") as file:
        pickle.dump(data, file)
    print(f"Model and features saved to {output_path}")

# Main workflow
if __name__ == "__main__":
    # Training image paths and labels
    image_paths = ["1.jpg", "2.jpg"]  # Replace with actual paths
    labels = ["A","B"]  # Replace with labels if available

    # Load pre-trained VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Extract features
    features = extract_features(image_paths, vgg_model)

    # Save the features and labels
    save_model_and_features(features, labels)
