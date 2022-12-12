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
import uuid

# Setup Logging
# Configure the logging system
logging.basicConfig(filename ='group2-final.log',
                    level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

asl_model = ''
hand_cropped_image = None
st_captured_image = None

threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
loaded_asl_model = None
loaded_asl_model_text = ''
image_shape = (224, 224, 3)
capture_button = None

def remove_background(frame, bgSubThreshold, learningRate):
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)    
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def preprocess_image(hand_crop, image_shape):
  """Preprocess Image
  Args:
      hand_crop (image): 
  """
  infer_image = cv2.resize(hand_crop,(image_shape[0],image_shape[1]))
  infer_image = np.expand_dims(infer_image, axis=0)
  return infer_image

def process_gesture(hand_cropped_image, asl_classes):
    logging.info('Using ASL Model' + asl_model)
    try:
        if not (hand_cropped_image is None):
            logging.info('Processing Gesture')
            logging.info('Saving Gesture into file')
            cv2.imwrite('hand_gesture.png',hand_cropped_image)
            data = preprocess_image(hand_cropped_image, image_shape)

            # ASL Model Predcition
            logging.info('Gesture Prediction')
            predicted_character = '1'
            predicted_character = predict_model(data, asl_model, asl_classes)

            return(predicted_character)
    except Exception as e:
        logging.error(str(e))


def draw_bounding_rect(image, brect):
  # Outer rectangle
  cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
    (0, 0, 0), 1)

  return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    # Pad the bounding rect
    x = x - 40
    y = y - 40
    w = w + 80
    h = h + 80
    

    return [x, y, x + w, y + h]

def predict_model(data, asl_model, asl_classes):
  global loaded_asl_model
  global loaded_asl_model_text
  asl_model = 'VGG1636'
  if asl_model != loaded_asl_model_text:
    logging.info('Loading Model ' + asl_model  )
    loaded_asl_model = tf.keras.models.load_model('./models/' + 'VGG1636' + '.h5')
    loaded_asl_model_text = asl_model
  predictions = loaded_asl_model.predict(data)
  print('Shape: {}'.format(predictions.shape))
  output_neuron = np.argmax(predictions[0])
  print('Most active neuron: {} ({:.2f}%)'.format(
      output_neuron,
      100 * predictions[0][output_neuron]
  ))
  logging.info('Predicted class:' +   str(asl_classes[output_neuron]))

  return str(asl_classes[output_neuron])


class OpenCVVideoProcessor(VideoProcessorBase):
    
    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.org = (50, 50)
        self.fontScale = 1
        self.color = (255, 0, 0)
        self.thickness = 2
        self.asl_classes = self.asl_classes_setup()


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
            _type_: Image with bounding rectangle
        """
        global hand_cropped_image
        # Setup media pipe capturing gesture    
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            model_complexity=0,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(image, hand_landmarks)
                    hand_cropped_image = image[brect[1]:brect[3], brect[0]:brect[2]]
                    image = draw_bounding_rect(image, brect)
                    predicted_character = process_gesture(hand_cropped_image, self.asl_classes)

                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (50,250)
                    fontScale              = 1
                    fontColor              = (255,255,255)
                    lineType               = 2

                    image = cv2.putText(image,'Predicted character:' + str(predicted_character),
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)

        return image

def asl_classes_setup():
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

def capture_gesture(image):
    """ Capture the ASL hand gesture

    Returns:
        _type_: Image with bounding rectangle
    """
    # Setup media pipe capturing gesture    
    mp_hands = mp.solutions.hands
    asl_classes = asl_classes_setup()

    with mp_hands.Hands(
        model_complexity=0,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(image, hand_landmarks)
                hand_cropped_image = image[brect[1]:brect[3], brect[0]:brect[2]]
                image = draw_bounding_rect(image, brect)
                predicted_character = process_gesture(hand_cropped_image, asl_classes)

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (50,250)
                fontScale              = 1
                fontColor              = (255,255,255)
                lineType               = 2

                image = cv2.putText(image,'Predicted character:' + str(predicted_character),
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

    return image

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = capture_gesture(img) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    img = cv2.putText(img, 'Group2 ASL Detection', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

########################## StreamLit Lay #############################################
# Title
tab1, tab2 = st.tabs(["Main", "Help"])

with tab1:
    st.title('AI GALLAUDET')
    st.header('Group-2 ASL Translator')
    st.subheader('Final Project MS AAI 521')
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints = {"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.write('Captured Image')
    captured_image = Image.open('./hand_gesture.png')
    st.image(captured_image)

with tab2:
    st.header("Group-2 Project Details")
    asl_characters = Image.open("./letters-numbers-ASL.png")
    st.image(asl_characters, width=500)
    transfer_traditional = Image.open("./tradition-vs-transfer-learning.jpg")
    st.image(transfer_traditional, width=500)
    cnn_image = Image.open("./cnn.jpg")
    st.image(cnn_image, width=500)


################################### SideBar #############################################
st.sidebar.write('Parameters')

# Using object notation
asl_model = st.sidebar.selectbox(
    "Model:",
    ("DenseNet12136", "ResNet10136", 
    "MobileNet36", "VGG1636", "pp_modelImage36"),
    index=3
)

with st.sidebar:
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    add_radio = st.radio(
        "Input Source:",
        ("Camera", "Image", "Video")
    )

    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
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

if __name__ == "__main__":
    logging.info('Starting application')
    logging.info("ASL Model:" + asl_model)