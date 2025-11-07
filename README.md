# Gesture-Tracker-with-MediaPipe

A real-time hand gesture recognition project built with **Python**, **OpenCV**, and **MediaPipe**.  

This app detects hand landmarks from a webcam feed and classifies common gestures:

- 👍 Thumbs Up  
- 👎 Thumbs Down  
- ✋ Open Palm (works palm-up & palm-down)  
- ✌️ Peace Sign  
- 👊 Fist  
- 👌 OK Sign  

---

## 🚀 Features
- Tracks **one or two hands** simultaneously
- Robust to **different hand orientations** (palm facing camera or flipped)
- Lightweight and runs in real-time on CPU
- Easy to extend with new gestures

---

## 🛠️ Tech Stack
- [Python 3](https://www.python.org/)  
- [OpenCV](https://opencv.org/) for video processing  
- [MediaPipe](https://developers.google.com/mediapipe) for hand tracking  

---

## 📦 Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/ivakehs77/Gesture-Tracker-with-MediaPipe.git
cd Gesture-Tracker-with-MediaPipe
pip install opencv-python mediapipe


Run the tracker:

python gesture.py

Press q to quit.

By default, the video feed is mirrored (like a selfie). You can disable this in the script with MIRROR = False.
