import streamlit as st
from cv2 import (VideoCapture, waitKey, imwrite, destroyAllWindows)
from cv2 import (convertScaleAbs, cvtColor, GaussianBlur, COLOR_BGR2GRAY)
from cv2 import (threshold, THRESH_BINARY)
from cv2 import (findContours, RETR_TREE,CHAIN_APPROX_SIMPLE)
from cv2 import convexHull, drawContours, contourArea
import pandas as pd
import numpy as np
from PIL import Image

st.title('ASL Alphabet Recognition')

# Capture Image
def capture_image():
    # initialize the camera
    # If you have multiple camera connected with 
    # current device, assign a value in cam_port 
    # variable according to that
    cam_port = 0
    cam = VideoCapture(cam_port)

    # reading the input using the camera
    result, image = cam.read()

    # If image will detected without any error, 
    # show result
    if result:

        # showing result, it take frame name and image 
        # output
        st.text("Image Captured")

        # saving image in local storage
        imwrite("hand_gesture.png", image)

    # If captured image is corrupted, moving to else part
    else:
        st.text("No image detected. Please! try again")

    cam.release()
    destroyAllWindows()

    return(image)

def contours_for_image(image):
    contours, hierarchy = findContours(image,RETR_TREE,CHAIN_APPROX_SIMPLE)
    #extract the largest contour
    max_area=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    cnt=contours[ci]
    #now draw the convex hull
    hull = convexHull(cnt)
    #displaying largest contour and convex hull
    drawing = np.zeros(image.shape,np.uint8)
    drawContours(drawing,[cnt],0,(0,255,0),2)
    drawContours(drawing,[hull],0,(0,0,255),2)
    st.image([drawing])

def brighten_image(image, amount):
    img_bright = convertScaleAbs(image, beta=amount)
    return img_bright

def gray_scale_image(image):
    gray_img = cvtColor(image, COLOR_BGR2GRAY)
    return gray_img

def blur_image(image, amount):
    blur_img = GaussianBlur(image, (11, 11), amount)
    return blur_img

def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)

    # Select sample or take picture
    asl_test_source = st.radio("ASL Image Test Source?",('camera','file'))
    # 0 for camera, 1 for file
    if asl_test_source == 'file': 
        image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
        if not image_file:
            return None
        original_image = Image.open(image_file)
    else:
        original_image = capture_image()

    
    original_image = np.array(original_image)
    # Convert image to gray scale
    processed_image = gray_scale_image(original_image)
    # Brighten image
    processed_image = brighten_image(processed_image, brightness_amount)
    # Blur Image
    processed_image = blur_image(processed_image, blur_rate )
    ret,processed_image = threshold(processed_image,125,255,THRESH_BINARY)
    contours_for_image(processed_image)

    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()

