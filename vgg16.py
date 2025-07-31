import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image

# Define model paths
model_paths = {
    "Model V1(Recommended)": 'ModelV3spliting2.h5',
    "Model V2": 'ModelV2Spliting1.h5'
}

# Sidebar
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model:", list(model_paths.keys()))

# Model Chosen
model_path = model_paths[model_choice]
model = load_model(model_path)

# Streamlit UI 
st.title("Brain Tumor Classification")
st.write("Upload an MRI image to classify it as Tumor or Non-Tumor:")

# Fungsi Prediksi 
def predict_tumor(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Logika Prediksi
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    if confidence > 0.5:
        class_label = "Tumor"
    else:
        class_label = "Non-Tumor"

    return class_label, confidence

uploaded_file = st.file_uploader("Choose an image...", type=["jpeg"])

if uploaded_file is not None:
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Classify the uploaded image
    class_label, confidence = predict_tumor(temp_file_path)

    # Display results
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    label_color = "#FF4B4B" if class_label == "Tumor" else "#4BFF88"
    st.markdown(
    f"""
    <div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;">
        The image is classified as <span style="color: {label_color};">{class_label}</span> 
        with <span style="color: {label_color};">{confidence:.2%}</span> confidence.
    </div>
    """,
    unsafe_allow_html=True
)
    # Display prediction on image using Matplotlib
    img = Image.open(temp_file_path)
    plt.imshow(img.resize((224, 224)))
    plt.title(f"Prediction: {class_label} ({confidence:.2%})")
    plt.axis('off')

    # Save the plot to a buffer
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Display plot in Streamlit
    st.pyplot(plt)

    # Download button to save the classified image with predictions
    st.download_button(
        label="Download Classified Image",
        data=buf,
        file_name="classified_image.png",
        mime="image/png"
    )

