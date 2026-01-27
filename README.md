# Automated Detection and Identification of Missing Person using AI

🚀 An AI-powered system to **detect and identify missing persons** using **face recognition** from images/video frames.  
This project helps reduce manual searching by automatically matching faces with stored records using deep learning based feature extraction and similarity matching.

---

## 🔥 Project Overview

Missing person identification is a real-world critical problem where manual identification from CCTV footage or crowd videos is highly time-consuming.  
This system automates the process by:

✅ Detecting faces from uploaded images/video frames  
✅ Extracting facial embeddings/features  
✅ Comparing with stored database faces  
✅ Returning the most similar match along with similarity score  

---

## ✨ Key Features

✅ Face detection and alignment using **Dlib**  
✅ Feature extraction using pre-trained deep learning models  
✅ Similarity matching using cosine similarity / distance score  
✅ Flask Web Application Interface  
✅ Can work with images extracted from videos (frame-by-frame processing)

---

## 🧠 AI/ML Techniques Used

- Face Detection  
- Facial Landmark Prediction  
- Feature Extraction (Embeddings)
- Similarity Matching (Cosine Similarity / Euclidean Distance)
- Classification / Identification pipeline

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---------|--------------------|
| Programming | Python |
| Web Framework | Flask |
| Image Processing | OpenCV |
| Face Recognition | dlib, face-recognition |
| Deep Learning | TensorFlow / Keras |
| Data Handling | NumPy, Pickle |
| Development | Jupyter Notebook / VS Code |

---

## 📂 Project Structure

Automated Detection and Identification of Missing Person using AI/
│── README.md
│── requirements.txt
│── .gitignore
│
├── src/
│ ├── app2.py # Flask application (main)
│ ├── main.py # Video to frames / supporting code
│
├── docs/
│ ├── Project Documentation.docx
│ ├── Project Final PPT.pptx
│ ├── plagarism checked journal.docx
│
├── screenshots/
│ ├── (add output screenshots here)
│
├── dataset/ # Optional (do not upload huge dataset)
│
└── models/ # Optional (not recommended to upload .dat / large files)


---

## 📌 Installation & Setup

### ✅ 1) Clone the Repository

```bash
git clone https://github.com/<your-username>/Missing-Person-Finder-AI.git
cd Missing-Person-Finder-AI

✅ 2) Create Virtual Environment (Recommended)
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Linux / Mac

source venv/bin/activate

✅ 3) Install Dependencies
pip install -r requirements.txt

▶️ Running the Project
✅ Run Flask Web App
python src/app2.py


Open in browser:

http://127.0.0.1:5000/

🧪 How It Works (Workflow)

1️⃣ User uploads an image/video frames
2️⃣ Face detected from input
3️⃣ Face embedding is extracted
4️⃣ Embedding is compared with stored database face vectors
5️⃣ Best match is returned with similarity score

📊 Results

✅ Successfully detects face
✅ Extracts unique facial features
✅ Performs similarity matching to identify possible missing person match
✅ Displays predicted result on web interface

📷 Screenshots (Add your output images here)

📌 Upload output images into /screenshots/ and update below:

🔹 Home Page

🔹 Detection Output

📁 Dataset

You can use:

Your own collected missing person face images dataset

Public face datasets (for testing)

📌 NOTE:
Do not upload large datasets directly into GitHub.
Use Google Drive links or Kaggle/Roboflow datasets and mention them here.

⚠️ Limitations

Face recognition accuracy reduces in low light / blurred frames

Performance depends on camera quality and face visibility

Side-face / occluded-face detection may reduce match confidence

Similar looking faces may cause false matches

🚀 Future Enhancements

✅ Live CCTV real-time stream integration
✅ Improved recognition using advanced models (ArcFace, FaceNet)
✅ Cloud deployment with scalable database
✅ Mobile app / Web dashboard integration
✅ Multi-face tracking and alert system

📄 Documentation

All project documentation is available in the /docs/ folder:

Project Report
Final PPT
Journal / References

👩‍💻 Author

Avanthika.K.S
B.E – Artificial Intelligence and Data Science
Avinashilingam Institute for Home Science and higher education for women, Coimbatore

⭐ Support

If you found this project useful, please ⭐ star the repository!
