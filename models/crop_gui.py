import cv2
import dlib
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class FaceDetector:
    """This class implements face detection using Dlib."""

    def __init__(self):
        # Initialize Dlib's face detector and shape predictor
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def detect_faces(self, image):
        """Detect faces in an image."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(grey, 1)
        return faces

def upload_and_process_image():
    """Function to upload an image, detect faces, and save cropped faces."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Could not read the image file.")
        return

    # Detect faces
    face_detector = FaceDetector()
    faces = face_detector.detect_faces(image)

    if not faces:
        messagebox.showinfo("No Faces Detected", "No faces were detected in the image.")
        return

    # Create directory to save cropped faces
    output_dir = "missing_person"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save cropped faces
    for idx, face in enumerate(faces):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cropped_face = image[max(0, y):min(image.shape[0], y + h), max(0, x):min(image.shape[1], x + w)]
        output_path = os.path.join(output_dir, f"face_{idx + 1}.jpg")
        cv2.imwrite(output_path, cropped_face)

    messagebox.showinfo("Success", f"Cropped faces saved in '{output_dir}' folder.")

    # Display the first detected face in the GUI
    show_cropped_face(output_path)

def show_cropped_face(image_path):
    """Display the cropped face in the GUI."""
    image = Image.open(image_path)
    image = image.resize((200, 200))
    photo = ImageTk.PhotoImage(image)

    img_label.config(image=photo)
    img_label.image = photo

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Face Detection and Cropping")
root.geometry("400x400")

# Add a label and button
label = tk.Label(root, text="Upload an image to detect and crop faces.", font=("Arial", 14))
label.pack(pady=20)

upload_button = tk.Button(root, text="Upload Image", command=upload_and_process_image, font=("Arial", 12), bg="blue", fg="white")
upload_button.pack(pady=10)

# Add a placeholder for displaying cropped faces
img_label = tk.Label(root)
img_label.pack(pady=20)

# Run the Tkinter main loop
root.mainloop()
