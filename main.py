import cv2
import dlib
import os

class FaceDetector:
    """This class implements face detection using Dlib."""

    def __init__(self):
        # Initialize Dlib's face detector and shape predictor
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def detect_faces_and_features(self, image):
        """Detect faces and facial features in an image using Dlib."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(grey, 1)
        detected_faces = []

        for face in faces:
            landmarks = self.shape_predictor(grey, face)

            # Check if key facial features are detected
            features_detected = {"eye": False, "nose": False, "mouth": False}

            # Eyes (landmarks 36-47)
            for i in range(36, 48):
                if 0 <= landmarks.part(i).x < image.shape[1] and 0 <= landmarks.part(i).y < image.shape[0]:
                    features_detected["eye"] = True

            # Nose (landmarks 27-35)
            for i in range(27, 36):
                if 0 <= landmarks.part(i).x < image.shape[1] and 0 <= landmarks.part(i).y < image.shape[0]:
                    features_detected["nose"] = True

            # Mouth (landmarks 48-67)
            for i in range(48, 68):
                if 0 <= landmarks.part(i).x < image.shape[1] and 0 <= landmarks.part(i).y < image.shape[0]:
                    features_detected["mouth"] = True

            detected_faces.append((face, features_detected))

        return detected_faces

def process_video(video_path):
    """Processes the video to detect faces and save cropped face images."""
    face_detector = FaceDetector()

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # Create directory to save cropped faces
    if not os.path.exists("cropped_faces"):
        os.makedirs("cropped_faces")

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces and facial features
        detected_faces = face_detector.detect_faces_and_features(frame)

        for idx, (face, features) in enumerate(detected_faces):
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Check if any feature is detected
            if any(features.values()):
                # Crop and save the face
                cropped_face = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
                face_filename = f"cropped_faces/frame{frame_count}_face{idx}.jpg"
                cv2.imwrite(face_filename, cropped_face)

        frame_count += 1

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "main2.mp4"  # Path to input video

    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Error: Shape predictor file not found. Ensure the file 'shape_predictor_68_face_landmarks.dat' is in the working directory.")
    else:
        process_video(video_path)
