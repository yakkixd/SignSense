# 🧠 SignSense: Real-Time Sign Language to Speech Converter

SignSense is a **real-time American Sign Language (ASL) recognition system** that translates hand gestures into **spoken English** using computer vision and speech synthesis.  
Built with **Python**, **MediaPipe**, **OpenCV**, and **pyttsx3**, it aims to bridge communication gaps between the hearing and speech-impaired communities and the world.

---

## 🌟 Features

- ✋ **Real-time hand detection and tracking** using [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe).
- 🔠 **ASL alphabet recognition** (A, B, C, D, F, L, O, V, Y).
- 🧮 **Custom rule-based classification** (optional support for ML-based `.pkl` models).
- 🗣️ **Speech synthesis (Text-to-Speech)** for detected signs.
- 🧾 **Sentence formation** with smooth gesture buffering and delay-based sign stability.
- 🖥️ **Interactive OpenCV interface** with FPS, confidence, and recognition info.
- 🎯 **Expandable** — easily train your own model for better accuracy.

---

## 🛠️ Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3.8+ |
| **Libraries** | OpenCV, MediaPipe, NumPy, pyttsx3, Pickle |
| **Model File** | `hand_landmarker.task` (MediaPipe Hand Detection Model) |
| **Classifier (Optional)** | `asl_classifier.pkl` (for custom ML models) |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yakkixd/SignSense.git
cd SignSense
```

### 2️⃣ Install Dependencies
```bash
pip install opencv-python mediapipe numpy pyttsx3
```

### 3️⃣ Add Model File
Place `hand_landmarker.task` in the root project directory (already included in this repo).

If you have a trained model (`asl_classifier.pkl`), put it in the same directory too.

---

## 🚀 Usage

Run the following command in your terminal:

```bash
python sign_language_converter.py
```

Once launched:

| Key | Action |
|-----|---------|
| **q** | Quit the application |
| **c** | Clear translated sentence |
| **s** | Speak the full sentence aloud |

---

## 🎥 Demo Interface

The webcam feed shows:
- **Detected hand landmarks**
- **Predicted ASL letter**
- **Confidence score**
- **Current FPS**
- **Sentence being formed**

Each recognized gesture is **added to the sentence buffer** if it remains stable for ~2.5 seconds.

---

## 🧩 System Workflow

1. **Capture Frame** → Webcam feed via OpenCV.  
2. **Detect Hand Landmarks** → MediaPipe Hand Landmarker (`.task` model).  
3. **Extract Features** → Finger distances, curl angles, relative coordinates.  
4. **Classify Sign** → Rule-based logic or trained classifier (`.pkl`).  
5. **Text-to-Speech** → Detected sign or full sentence spoken aloud.  

---

## 🧠 Model & Recognition Details

Currently recognizes these **ASL static letters**:
> A, B, C, D, F, L, O, V, Y

Letters **J** and **Z** (motion-based) are not supported yet.

### Want to improve accuracy?
- Use `collect_training_data()` (to be added) to gather your own dataset.
- Train a model and save it as `asl_classifier.pkl`.

---

## 🧰 File Structure

```
SignSense/
│
├── sign_language_converter.py    # Main program
├── hand_landmarker.task          # MediaPipe model for hand detection
└── README.md                     # Project documentation
```

---

## 🗣️ Future Scope

- Support for **dynamic gestures** (e.g., J, Z, words).  
- Integration with **speech recognition** for two-way interaction.  
- GUI version for easier accessibility.  
- Cloud API for web-based deployment.  

---

