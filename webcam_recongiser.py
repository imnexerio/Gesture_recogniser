# Imports necessary modules.
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
GestureRecognizer = mp.tasks.vision.GestureRecognizer

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# Create a GestureRecognizer object with additional parameters.
num_hands = 1  # You can adjust this value based on your requirement.
min_hand_detection_confidence = 0.8  # Adjust as needed.
min_hand_presence_confidence = 0.3  # Adjust as needed.
min_tracking_confidence = 0.5  # Adjust as needed.

# Create a canned gesture options object.
# canned_gestures_options = solutions.hands.HandLandmarkModelOptions()

model_path = os.path.abspath("AIMS\\drone_model9794.task")


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=num_hands,
    min_hand_detection_confidence=min_hand_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
)
recognizer=GestureRecognizer.create_from_options(options)
# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


while True:
    # Read a new frame.
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB and then to uint8.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # Create a mediapipe Image.
    image = mp.Image(mp.ImageFormat.SRGB, frame_rgb)

    # Run gesture recognition.
    recognition_result = recognizer.recognize(image)

    annotated_image = draw_landmarks_on_image(image.numpy_view(), recognition_result)

    cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # lm.x, lm.y, lm.z are the landmark positions
    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), recognition_result)

    
        
    try:
        top_gesture = recognition_result.gestures[0][0]
        gesture_text = f"Gesture recognized: {top_gesture.category_name} ({(top_gesture.score)*100})"
        print(gesture_text)
        
    except:
        gesture_text = "No gesture recognized."
        print(gesture_text)
    
    cv2.putText(annotated_image, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # Quit if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows when done.
cap.release()
cv2.destroyAllWindows()
