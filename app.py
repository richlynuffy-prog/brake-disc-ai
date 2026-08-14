import streamlit as st
import tensorflow as tf 
from PIL import Image
import numpy as np

#SET PAGE CONFIGURATION
st.set_page_config(
    page_title = "Brake Disc AI Inspector",
    page_icon = "🚙",
    layout = "centered"
)
#LOAD THE TRAINED MODEL
model = tf.keras.models.load_model("brake_disc_model.keras")

#CLASS NAMES
class_names = ["bad", "good", "intermediate"]

#CREATE THE PREDICTION CODE
def predict_image(image):

    #RESIZE THE IMAGE
    image =image.resize((224, 224))

    #CONVERT TO NUMPY ARRAY
    image = np.array(image)

    #ADD BATCH DIMENSION
    image = np.expand_dims(image, axis=0)

    #MAKE PREDICTION
    prediction = model.predict(image, verbose=0)

    #FIND CLASS WITH HIGHEST PROBABILITY
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    #CONFIDENCE
    confidence = prediction[0][predicted_index]*100

    #GET PROBABILITY OF EVERY CLASS
    probabilities = prediction[0]*100
    return predicted_class, confidence, probabilities

#STREAMLIT INTERFACE
st.title("🚙Brake Disc AI Inspector")
st.subheader("AI-Based Brake Disc Condition Detection")
st.write("Upload an image of the brake disc for classification")
st.divider()

#UPLOADER KEY
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
#IMAGE UPLOADER
uploaded_file = st.file_uploader("Upload a brake disc image", type =["jpg", "jpeg", "png"])

#CLEAR IMAGE WITHOUT RESTARTING APP
if st.button("Clear / New Image"):
    st.session_state.uploader_key += 1
    st.rerun()

#DISPLAY THE UPLOADED IMAGE
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image,
             caption = "Uploaded Brake Disc",
             width = 400
             )
    if st.button("Analyze Brake Disc"):
        predicted_class, confidence, probabilities = predict_image(image)
        st.subheader("Prediction")
        st.write(f"Condtion: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        #CONFIDENCE WARNIG
        if confidence <= 70:
            st.warning("!!! Low confidence prediction!!!.  "
                       "please provide a clearer image and make sure only a single brake disc is in the image")
        elif confidence >= 71 <= 80:
            st.info("!!!Moderate confidence prediction!!!.  "
                    "further inspection is recommended")
        else:
            st.success("!!!High probability Prediction!!!.  "
                       "if image doesnt match prediction, provide a clearer image")

        #CLASS PROBABILITIES
        st.subheader("class probabilities")

        #GOOD
        st.write(f"Good -- {probabilities[1]:.2f}%")
        st.progress(float(probabilities[1] / 100))

        #INTERMEDIATE
        st.write(f"Intermediate -- {probabilities[2]:.2f}%")
        st.progress(float(probabilities[2] / 100))

        #BAD
        st.write(f"Bad -- {probabilities[0]:.2f}%")
        st.progress(float(probabilities[0] / 100))