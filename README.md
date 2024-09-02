# Real-time Hand Sign Language Detection

## Overview

This project aims to enable real-time gesture recognition within video streams for effective sign language interpretation. The Sign Language Action Detection system achieves an impressive accuracy of 92%, utilizing a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.

## Technologies Used

1. OpenCV
2. Mediapipe
3. TensorFlow
4. Matplotlib
5. Scikit Learn

## Deep Learning Networks

1. CNN
2. LSTM

## Workflow

1. **Data Collection:**
   - Utilized OpenCV and the OS module in Python to collect training data samples from the front camera.
   - Data samples were labeled and organized by folder name for easy management.

2. **Holistic Model:**
   - Employed Mediapipe's holistic model to draw a mesh around the body in the captured frames.

3. **Frame Sequencing:**
   - For each input, 30 frames were used to form a sequence.
   - Three different hand gestures were targeted: "hello," "thanks," and "I love you."
   - Each of the frame is stored as a numpy array.

4. **Data Splitting:**
   - Employed scikit-learn to split the data into training and testing sets.

5. **Model Architecture:**
   - Constructed a sequential model with 3 LSTM layers followed by 3 Dense layers.
   - The model architecture consists of a total of 596,675 parameters.

6. **Training:**
   - Trained the model for 200 epochs to ensure robust learning.
   - Weights were saved for future use.

7. **Real-time Prediction:**
   - Utilized the trained model to make predictions in real-time using OpenCV.
   - The real-time prediction enables effective interpretation of hand sign language gestures within video streams.

## Getting Started

To use the real-time hand sign language detection model, follow these steps:

1. Clone the repository.
2. Install the required dependencies (OpenCV, Mediapipe, TensorFlow, Matplotlib).
3. Run the provided scripts to capture and process real-time video streams.

![WhatsApp Video 2024-01-28 at 02 58 35_b89b4869](https://github.com/Taha0229/realtime-sign-lang-detection/assets/113607983/9ee61584-5e15-4c64-abe1-cece1e55e1af)
