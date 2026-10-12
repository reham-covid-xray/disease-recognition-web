import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Medical AI System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f4f8fc;
}

[data-testid="stSidebar"] {
    background-color: #102a43;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    color: #123b5d;
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 8px;
}

.subtitle {
    color: #627d98;
    text-align: center;
    font-size: 17px;
    margin-bottom: 30px;
}

.section-title {
    color: #123b5d;
    font-size: 27px;
    font-weight: bold;
}

.info-card {
    background-color: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #d9e6f2;
    margin-bottom: 15px;
}

div.stButton > button {
    background-color: #1677b8;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #105985;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.markdown("# 🩺 Medical AI")
st.sidebar.write("Medical Image Recognition")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "X-ray Analysis", "About System"]
)

st.sidebar.markdown("---")
st.sidebar.caption("AI Medical Research Project")

# =========================
# Home Page
# =========================
if page == "Home":

    st.markdown(
        '<div class="main-title">'
        'Medical Image Disease Recognition System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Chest X-ray Classification Using CNN'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Welcome to our medical image recognition system. "
        "This project demonstrates how Artificial Intelligence "
        "can classify chest X-ray images."
    )

    st.markdown('<div class="section-title">Our System</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🩻 Chest X-ray Analysis</h3>
            <p>Upload a chest X-ray image and let the trained
            AI model analyze it.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 AI Classification</h3>
            <p>The system classifies images into the model's
            two categories: NORMAL and COVID-19.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project Objectives</div>',
                unsafe_allow_html=True)

    st.markdown("""
    - Apply Artificial Intelligence to medical image classification.
    - Demonstrate the use of a Convolutional Neural Network (CNN).
    - Build a simple, accessible web interface for image analysis.
    - Present model predictions clearly to the user.
    """)

    st.warning(
        "Research and educational use only. "
        "This system is not a substitute for professional medical advice."
    )

# =========================
# X-ray Analysis Page
# =========================
elif page == "X-ray Analysis":

    st.markdown(
        '<div class="main-title">Chest X-ray Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload an image to run the AI classification model'
        '</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "This experimental model cannot confirm or rule out a disease. "
        "Always consult a qualified healthcare professional."
    )

    uploaded_file = st.file_uploader(
        "Choose a Chest X-ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(
                    image,
                    caption="Chest X-ray",
                    use_container_width=True
                )

            with col2:
                st.subheader("Image Information")
                st.write(f"Width: {image.width} pixels")
                st.write(f"Height: {image.height} pixels")
                st.write("Format: RGB")

            if st.button("Analyze Image", use_container_width=True):

                with st.spinner("Analyzing image..."):

                    @st.cache_resource
                    def load_model():
                        interpreter = tf.lite.Interpreter(
                            model_path="covid_normal_model_small.tflite"
                        )
                        interpreter.allocate_tensors()
                        return interpreter

                    interpreter = load_model()

                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()

                    img = image.resize((224, 224))
                    img_array = np.array(
                        img, dtype=np.float32
                    ) / 255.0

                    img_array = np.expand_dims(img_array, axis=0)

                    expected_shape = input_details[0]["shape"]
                    if list(img_array.shape) != list(expected_shape):
                        st.error(
                            "The uploaded image does not match "
                            "the model's expected input shape."
                        )
                        st.stop()

                    interpreter.set_tensor(
                        input_details[0]["index"],
                        img_array
                    )

                    interpreter.invoke()

                    prediction = interpreter.get_tensor(
                        output_details[0]["index"]
                    )[0][0]

                st.markdown("---")
                st.subheader("Classification Result")

                # Keep the same threshold as the original application.
                if prediction >= 0.5:
                    result = "NORMAL"
                else:
                    result = "COVID-19"

                if result == "NORMAL":
                    st.success(f"Model prediction: {result}")
                else:
                    st.error(f"Model prediction: {result}")

                st.caption(
                    f"Raw model output: {float(prediction):.4f}"
                )

                st.info(
                    "This is an experimental AI prediction, not a diagnosis. "
                    "The model output is not necessarily a calibrated "
                    "probability. Please consult a medical professional."
                )

        except Exception as e:
            st.error("An error occurred while processing the image.")
            st.exception(e)

    else:
        st.markdown("""
        <div class="info-card">
            <h3>How to use</h3>
            <p>1. Select a chest X-ray image.</p>
            <p>2. Review the uploaded image.</p>
            <p>3. Click Analyze Image to view the model output.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# About System Page
# =========================
elif page == "About System":

    st.markdown(
        '<div class="main-title">About the System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Learn about the project, its objectives, and technologies'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Project Overview</div>',
                unsafe_allow_html=True)

    st.write("""
    Medical Image Disease Recognition System is an educational
    project that demonstrates the application of Artificial Intelligence
    in classifying chest X-ray images.

    The current model supports two output classes: NORMAL and COVID-19.
    Its predictions should not be used to make medical decisions.
    """)

    st.markdown('<div class="section-title">Technologies Used</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🐍 Python</h4>
            <p>Programming language used to develop the application.</p>
        </div>
        <div class="info-card">
            <h4>🧠 TensorFlow Lite</h4>
            <p>Runs the optimized machine learning model.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🌐 Streamlit</h4>
            <p>Creates the interactive web application.</p>
        </div>
        <div class="info-card">
            <h4>🖼️ Pillow and NumPy</h4>
            <p>Process images and prepare numerical input for the model.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">How It Works</div>',
                unsafe_allow_html=True)

    st.markdown("""
    1. The user uploads a chest X-ray image.
    2. The image is resized to 224 × 224 pixels.
    3. Pixel values are normalized to the range 0–1.
    4. The image is passed to the TensorFlow Lite model.
    5. The application displays the class selected by the model.
    """)

    st.markdown("---")

    st.caption(
        "Medical Image Disease Recognition System | "
        "Educational and Research Project"
    )