import streamlit as st

# Title
st.title('AI GALLAUDET')
st.header('Group-2 ASL Translator')
st.subheader('Final Project MS AAI 521')
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


st.text('Background')
st.text('Desgin')
st.text('Experimentation')
st.text('Results')
# Object notation
st.sidebar.write('Model Experimentation')

# "with" notation
with st.sidebar:
    st.sidebar.button('Run')

import streamlit as st

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Select Model to use?",
    ("Inception", "Dense", "XCeption", "Inception-ResNetV2", 
    "SENet", "NASNET", "EfficientNetB2", "EfficientNetB4", "EfficientNetB6")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Train or Predict",
        ("Train", "Predict")
    )