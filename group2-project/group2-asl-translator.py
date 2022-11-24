import streamlit as st
import mediapipe as mp
import numpy as np
from utils import *
import cv2
import logging
import sys
import av
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration,WebRtcMode,webrtc_streamer
import tensorflow as tf
from PIL import Image

# Setup Logging
# Configure the logging system
logging.basicConfig(filename ='group2-final.log',
                    level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

asl_model = ''
st_captured_image = None

def preprocess_image(hand_crop, image_shape):
  """Preprocess Image
  Args:
      hand_crop (image): 
  """
  hand_crop = cv2.resize(hand_crop,(image_shape[0],image_shape[1]))
  #cv2.imwrite('./hand_gesture_cropped.png', hand_crop)
  data = np.empty((1,image_shape[0],image_shape[1],image_shape[2]))
  data[0] = hand_crop
  return data

def predict_model(data, model, asl_classes):
  predictions = model.predict(data)
  print('Shape: {}'.format(predictions.shape))
  output_neuron = np.argmax(predictions[0])
  print('Most active neuron: {} ({:.2f}%)'.format(
      output_neuron,
      100 * predictions[0][output_neuron]
  ))
  logging.info('Predicted class:' +   str(asl_classes[output_neuron]))

  return str(asl_classes[output_neuron])

def about_project(st):
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown('''
        # About Group-2 Project\n 
        In this application we are using **MediaPipe** for gesture capture. **StreamLit** is to create the Web Graphical User Interface (GUI) 
        ''')

def setup_side_bar_layout():
    # Object notation
    st.sidebar.write('Parameters')

    # Using object notation
    asl_model = st.sidebar.selectbox(
        "Model:",
        ("Inception", "Dense", "XCeption", "Inception-ResNetV2", 
        "SENet", "NASNET", "EfficientNetB2", "EfficientNetB4", "EfficientNetB6"),
        index=6
    )
    
    with st.sidebar:
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # Using "with" notation
    with st.sidebar:
        add_radio = st.radio(
            "Input Source:",
            ("Camera", "Image", "Video")
        )
    
    with st.sidebar:
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    
    with st.sidebar:
        about_project(st) 

def setup_main_layout():
    # Title
    st.title('AI GALLAUDET')
    st.header('Group-2 ASL Translator')
    st.subheader('Final Project MS AAI 521')
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

def setup_captured_image():
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.write('Captured Image')
    captured_image = Image.open('./hand_gesture.png')
    st_captured_image = st.image(captured_image)

def main():
    setup_main_layout()
    setup_side_bar_layout()
    sign_language_detector()
    setup_captured_image()

def sign_language_detector():
    
    class OpenCVVideoProcessor(VideoProcessorBase):
        
        def __init__(self) -> None:
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.org = (50, 50)
            self.fontScale = 1
            self.color = (255, 0, 0)
            self.thickness = 2
            self.asl_classes = self.asl_classes_setup()
            self.model_name = ''
            self.model = None
            self.model_setup()


        def asl_classes_setup(self):
            digit_0 = '0'
            digit_classes = []    
            digit_classes = [(chr(ord(digit_0)+i)) for i in range(10)]
            letter_a ='a'
            letter_classes = []
            # starting from the ASCII value of 'a' and keep increasing the 
            # value by i.
            letter_classes =[(chr(ord(letter_a)+i)) for i in range(26)]
            asl_classes = digit_classes + letter_classes
            return asl_classes

        def model_setup(self):
            if self.model_name != asl_model:
                self.model = tf.keras.models.load_model('models/' + asl_model + '.h5')
                self.model_name = asl_model

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            # Using cv2.putText() method
            img = self.capture_gesture(img) 
            img = cv2.putText(img, 'Group2 ASL Detection', self.org, self.font, 
                            self.fontScale, self.color, self.thickness, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


        def capture_gesture(self, image):
            """ Capture the ASL hand gesture

            Returns:
                _type_: Cropped Hand Gesture image
            """
            # Setup media pipe capturing gesture    
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_hands = mp.solutions.hands

            hand_found = False
            hand_crop = None
            image_shape = (50, 50, 3)
            with mp_hands.Hands(
                model_complexity=0,
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
                h, w, c = image.shape
                results = hands.process(image)
                if results.multi_hand_landmarks:
                    logging.info('Found hands')
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get bounding rectangle    
                        x_max = 0
                        y_max = 0
                        x_min = w
                        y_min = h
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            if x > x_max:
                                x_max = x + 200
                            if x < x_min:
                                x_min = x - 200
                            if y > y_max:
                                y_max = y + 200 
                            if y < y_min:
                                y_min = y - 200 
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    hand_crop = image[y_min:y_max, x_min:x_max]
                    hand_found = True
                    if st_captured_image:
                        captured_image = Image.fromarray(cv2.resize(hand_crop,(100,100)))
                        st_captured_image.empty()
                        st_captured_image.image(captured_image)
                    #cv2.imwrite('./hand_gesture.png', hand_crop)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                try:
                    if hand_found and not (hand_crop is None):
                        data = preprocess_image(hand_crop, image_shape)

                        # ASL Model Predcition
                        logging.info('Gesture Prediction')
                        predicted_character = '1'
                        predicted_character = predict_model(data, self.model, self.asl_classes)

                        font                   = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (50,250)
                        fontScale              = 1
                        fontColor              = (255,255,255)
                        lineType               = 2

                        image = cv2.putText(image,'Predicted character:' + predicted_character,
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
                except Exception as e:
                    logging.error(str(e))


            return image


    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints = {"video": True, "audio": False},
        async_processing=True,
    )


if __name__ == "__main__":
    main()