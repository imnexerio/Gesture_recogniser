# Gesture Recognition with Mediapipe Model Maker

## Overview
Gesture Recognition with Mediapipe Model Maker is a project that utilizes the [Mediapipe Model Maker](https://github.com/google/mediapipe) library for training a gesture recognition model. The trained model can recognize gestures in images, and a live webcam detection script allows real-time gesture recognition through a webcam feed.

The project is specifically designed to recognize gestures related to rock-paper-scissors, using a dataset of hand images in various poses corresponding to these gestures.

## Table of Contents
- [Requirements](#opencv,numpy,mediapipe,mediapipe_model_maker)
- [Training](#get a set of hand gestures)
- [Live Webcam Detection](#live-webcam-detection)

## Requirements
- Python 3.11.x
- [Mediapipe Model Maker](https://github.com/google/mediapipe)

## Training
- **Dataset Preparation:**
  - Organize your rock-paper-scissors hand gesture images in the specified directory (`/content/drive/MyDrive/hand data_compressed` in this example).
- **Hyperparameter Configuration:**
  - Adjust hyperparameters in the training script (`train_gesture_model.py`) such as learning rates, batch sizes, and epochs.
- **Run the Training Script:**
    ```bash
    python train_gesture_model.py
    ```

## Live Webcam Detection
- **Adjust Parameters:**
  - Fine-tune parameters in the live detection script (`live_detection.py`) if necessary.

## Results
- **Hyperparameter Tuning:**
  - The best hyperparameters and test results are stored in `hyperparameter_results`.
- **Visualize Results:**
  - Examine the generated plots to visualize the performance of different hyperparameter combinations.

## Contributing
Feel free to contribute to this project! If you find issues, have suggestions for improvements, or want to add features, please open an issue or submit a pull request.

