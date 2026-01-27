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
## 📁 Project Structure

```bash
Automated Detection and Identification of Missing Person using AI/
│── README.md
│── requirements.txt
│── .gitignore
│
├── src/
│   ├── app2.py               # Flask application (main)
│   ├── main.py               # Video to frames / supporting code
│
├── docs/
│   ├── Project Documentation.docx
│   ├── Project Final PPT.pptx
│   ├── plagarism checked journal.docx
│
├── screenshots/
├── dataset/                  
└── models/                 


```
## 📌 Installation & Setup

### ✅ 1) Clone the Repository

```bash
git clone https://github.com/<your-username>/Missing-Person-Finder-AI.git
cd Missing-Person-Finder-AI
```

### ✅ 2) Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

### ✅ 3) Activate Virtual Environment

**Windows**
```bash
venv\Scripts\activate
```

**Linux / Mac**
```bash
source venv/bin/activate
```

### ✅ 4) Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### ✅ Run Flask Web App

```bash
python src/app2.py
```

Open in browser:

```text
http://127.0.0.1:5000/
```

---

## 🧪 How It Works (Workflow)

1️⃣ User uploads an image / video frames  
2️⃣ Face is detected from the input  
3️⃣ Facial embedding/features are extracted  
4️⃣ Features are compared with stored database faces  
5️⃣ Best match is returned with similarity score  

---

## 📊 Output / Results

✅ Successfully detects faces  
✅ Extracts unique facial embeddings  
✅ Matches with stored face vectors  
✅ Displays predicted missing person match (if found)  

---

## 📷 Screenshots

### 🔹 Home Page
```md
![Home Page](screenshots/home.png)
```

### 🔹 Result / Match Output
```md
![Result Page](outputs/Screenshot 2025-02-12 111528.png)
```

---

## 📁 Dataset

You can use:
- Your own collected missing person face images dataset  
- Public face datasets (only for testing)

📌 NOTE:  
Do not upload large datasets directly into GitHub.  
Instead, provide a Google Drive / Kaggle link.

---

## ⚠️ Limitations

- Face recognition accuracy reduces in low light / blurred frames  
- Performance depends on camera quality and face visibility  
- Occlusions (mask, cap) can reduce recognition score  
- Similar-looking faces may lead to false matches  

---

## 🚀 Future Enhancements

✅ Real-time CCTV/live stream integration  
✅ Improve accuracy with models like ArcFace / FaceNet  
✅ Cloud deployment with scalable face database  
✅ Mobile/Web dashboard with alerts  
✅ Multi-face tracking and notifications  

---

## 📄 Documentation

All documentation is available in the `/docs/` folder:

- Project Documentation (Report)  
- Final PPT  
- Journal / Reference document  

---

## 👩‍💻 Author

**Avanthi**  
M.Tech – Artificial Intelligence Engineering (AIE)  
Amrita Vishwa Vidyapeetham, Coimbatore  

---

## ⭐ Support

If you found this project useful, please ⭐ star the repository!
