import cv2
import dlib
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

class FaceDetector:
    """This class implements face detection using Dlib."""

    def __init__(self):
        # Initialize Dlib's face detector and shape predictor
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    def detect_and_compute_embedding(self, image):
        """Detect faces and compute embeddings."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(grey, 1)
        embeddings = []

        for face in faces:
            shape = self.shape_predictor(grey, face)
            embedding = np.array(self.face_recognizer.compute_face_descriptor(image, shape))
            embeddings.append((face, embedding))

        return embeddings

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
        return 1
    else:
        print("No similar image found.")
        return 0
def compare_faces(uploaded_embedding, video_embeddings, threshold=0.6):
    """Compare uploaded face embedding with video embeddings."""
    for frame_idx, (face, embedding) in enumerate(video_embeddings):
        similarity = cosine_similarity([uploaded_embedding], [embedding])[0][0]
        if similarity > threshold:
            return frame_idx, face, similarity
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle form submission
        name = request.form["name"]
        age = request.form["age"]
        address = request.form["address"]
        parents_name = request.form["parents_name"]

        # Save uploaded face image
        uploaded_file = request.files["face_image"]
        uploaded_path = "static/uploaded_face.jpg"
        uploaded_file.save(uploaded_path)

        # Load and process uploaded face
        detector = FaceDetector()
        uploaded_image = cv2.imread(uploaded_path)
        uploaded_embeddings = detector.detect_and_compute_embedding(uploaded_image)

        if not uploaded_embeddings:
            return "No face detected in uploaded image. Please try again."

        uploaded_embedding = uploaded_embeddings[0][1]

        # Process video
        video_path = "raganew.mp4"
        video_capture = cv2.VideoCapture(video_path)
        frame_idx = 0
        match = None

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            embeddings = detector.detect_and_compute_embedding(frame)
            for face, embedding in embeddings:
                
                similarity = cosine_similarity([uploaded_embedding], [embedding])[0][0]
                vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

                # Load saved features and labels
                saved_features, saved_labels = load_saved_model()

                # Check similarity
                gh=check_similarity(uploaded_path, vgg_model, saved_features, saved_labels)
                if similarity > 0.95 or gh==1:  # Threshold
                    match = (frame_idx, face, similarity)
                    break

            if match:
                break

            frame_idx += 1

        video_capture.release()

        if match:
            frame_idx, face, similarity = match
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            match_frame_path = f"static/match_frame_{frame_idx}.jpg"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(match_frame_path, frame)
            match_results = [{"frame": frame_idx, "similarity": similarity, "image": match_frame_path}]
            return render_template("results.html", matches=match_results)
        else:
            return "No match found in the video."

    return render_template("index.html")

if __name__ == "__main__":
    if not os.path.exists("shape_predictor_68_face_landmarks.dat") or not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
        print("Error: Required Dlib model files not found. Ensure 'shape_predictor_68_face_landmarks.dat' and 'dlib_face_recognition_resnet_model_v1.dat' are in the working directory.")
    else:
        app.run(debug=True)
