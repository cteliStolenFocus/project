# For webcam input:

import mediapipe as mp
import pandas as pd
import seaborn as sns
import numpy as np
import os
import cv2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions

def capture_gesture():
  """ Capture the ASL hand gesture

  Returns:
      _type_: Cropped Hand Gesture image
  """
  # Setup media pipe capturing gesture    
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      h, w, c = image.shape
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      print(results.multi_hand_world_landmarks)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          # Get bounding rectangle    
          x_max = 0
          y_max = 0
          x_min = w
          y_min = h
          for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x > x_max:
              x_max = x + 25
            if x < x_min:
              x_min = x - 25 
            if y > y_max:
              y_max = y + 25
            if y < y_min:
              y_min = y - 25 
          cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
          hand_crop = image[y_min:y_max, x_min:x_max]
          hand_crop = cv2.flip(hand_crop,1)
          cv2.imwrite('./hand_gesture.png', hand_crop)
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
      key = cv2.waitKey(5)
      # ESC or 'q'
      if key == 27 or key == 113:
        cv2.destroyAllWindows()
        break
  cap.release()

  return hand_crop

def data_processing():
  """ Process Data for training and data visualization
  """      
  path = './asl_dataset'
  names = []
  nums = []
  data = {'Name of class':[],'Number of samples':[]}

  for i in os.listdir(path):
      nums.append(len(os.listdir(path+'/'+i)))
      names.append(i)

  data['Name of class']+=names
  data['Number of samples']+=nums

  df = pd.DataFrame(data)
  sns.barplot(x=df['Name of class'],y=df['Number of samples'])

def preprocess_image(hand_crop):
  """Preprocess Image
  Args:
      hand_crop (image): 
  """
  hand_crop = cv2.resize(hand_crop,(224,224))
  cv2.imwrite('./hand_gesture_cropped.png', hand_crop)
  data = np.empty((1,224,224,3))
  data[0] = hand_crop
  data = preprocess_input(data)
  return data

def train_model():
  model = MobileNetV2(weights='imagenet')
  return model

model = train_model()
# model.summary()  # Uncoomment this to print a long summary!
hand_crop = capture_gesture()
data = preprocess_image(hand_crop)
predictions = model.predict(data)
print('Shape: {}'.format(predictions.shape))

output_neuron = np.argmax(predictions[0])
print('Most active neuron: {} ({:.2f}%)'.format(
    output_neuron,
    100 * predictions[0][output_neuron]
))

for name, desc, score in decode_predictions(predictions)[0]:
    print('- {} ({:.2f}%%)'.format(desc, 100 * score))
