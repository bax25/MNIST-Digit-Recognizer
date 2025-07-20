import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Configure page
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢", layout="wide")

# Initialize session state for canvas clearing
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_0"


@st.cache_resource
def load_model():
    """Load Keras model with error handling"""
    try:
        model = tf.keras.models.load_model("mnist_cnn_model_enhanced.keras")
        return model
    except FileNotFoundError:
        try:
            model = tf.keras.models.load_model("mnist_cnn_model_enhanced.h5")
            return model
        except FileNotFoundError:
            st.error("Model file not found! Please train the model first.")
            st.info(
                "Expected files: 'mnist_cnn_model_enhanced.keras' or 'mnist_cnn_model_enhanced.h5'"
            )
            st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


def preprocess_image(img_data):
    """Enhanced image preprocessing without color inversion"""
    try:
        if len(img_data.shape) == 3:
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_data

        if np.sum(img_gray) == 0:
            return None, None

        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        img_reshaped = img_resized.reshape(1, 28, 28, 1)
        img_normalized = img_reshaped.astype(np.float32) / 255.0

        return img_normalized, img_resized

    except Exception as e:
        return None, None


def make_prediction(model, img_data):
    """Make prediction using TensorFlow/Keras directly"""
    try:
        prediction = model.predict(img_data, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        return predicted_digit, confidence, prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None


def clear_canvas():
    """Clear canvas and reset all related states"""
    st.session_state.canvas_key = f"canvas_{int(time.time() * 1000)}"
    # Clear any stored predictions or processed images
    if "prediction_results" in st.session_state:
        del st.session_state.prediction_results


# Load model
model = load_model()

# Header
st.title("🔢 Enhanced MNIST Digit Recognizer (TensorFlow)")

# Instructions
st.markdown(
    """
**Instructions:**
1. Draw a digit (0-9) on the canvas below
2. Use white stroke on black background for best results
3. Click 'Predict' to see the AI's prediction
4. Click 'Clear Canvas' to start over
"""
)

# Main layout - Three columns
main_col1, main_col2, main_col3 = st.columns([1, 1, 1.2])

# COLUMN 1: Drawing Canvas
with main_col1:
    st.subheader("✏️ Drawing Canvas")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,  # Use dynamic key for clearing
    )

    # Action buttons
    st.subheader("Controls")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        predict_btn = st.button("🔮 Predict Digit", type="primary")

    with col_btn2:
        if st.button("🗑️ Clear Canvas"):
            clear_canvas()
            st.rerun()

# COLUMN 2: Preprocessed Image Preview
with main_col2:
    st.subheader("🔍 AI Model Input")

    # Only show preview if canvas has data and something is drawn
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:

        processed_img, preprocessed_display = preprocess_image(
            canvas_result.image_data.astype("uint8")
        )

        if processed_img is not None and preprocessed_display is not None:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(preprocessed_display, cmap="gray")
            # ax.set_title("28×28 Input Image", fontsize=5, fontweight="bold")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

            # Show image statistics
            st.write("**Image Stats:**")
            st.write("- Size: 28×28 pixels")
            st.write("- Range: 0.0 - 1.0")
            st.write(f"- Mean: {np.mean(preprocessed_display):.3f}")
        else:
            st.info("Draw on canvas to see preview")
    else:
        st.info("Canvas is empty")

# COLUMN 3: Prediction Results
with main_col3:
    st.subheader("🎯 Prediction Results")

    if predict_btn:
        if (
            canvas_result.image_data is not None
            and np.sum(canvas_result.image_data) > 0
        ):

            processed_img, preprocessed_display = preprocess_image(
                canvas_result.image_data.astype("uint8")
            )

            if processed_img is not None:
                with st.spinner("AI is thinking..."):
                    predicted_digit, confidence, probabilities = make_prediction(
                        model, processed_img
                    )

                if predicted_digit is not None:
                    # Store results in session state to persist
                    st.session_state.prediction_results = {
                        "digit": predicted_digit,
                        "confidence": confidence,
                        "probabilities": probabilities,
                    }

                    # Main prediction result
                    st.success(f"**Predicted Digit: {predicted_digit}**")
                    st.info(f"**Confidence: {confidence:.1f}%**")

                    # Confidence indicator
                    if confidence >= 85:
                        st.success("🎯 Excellent Confidence!")
                    elif confidence >= 70:
                        st.success("✅ High Confidence")
                    elif confidence >= 50:
                        st.warning("⚠️ Medium Confidence")
                    else:
                        st.error("❌ Low Confidence")

                    # Probability distribution
                    st.subheader("📊 All Digit Probabilities")
                    prob_data = {
                        f"Digit {i}": prob for i, prob in enumerate(probabilities)
                    }
                    st.bar_chart(prob_data)

                    # Show top 3 predictions
                    top_3_idx = np.argsort(probabilities)[-3:][::-1]
                    st.write("**🏆 Top 3 Predictions:**")
                    medals = ["🥇", "🥈", "🥉"]
                    for i, idx in enumerate(top_3_idx):
                        st.write(f"{medals[i]} Digit {idx}: {probabilities[idx]:.1%}")

            else:
                st.warning("⚠️ Unable to process the image. Please try drawing again.")
        else:
            st.warning("📝 Please draw something on the canvas first!")

    # Show previous results if they exist
    elif "prediction_results" in st.session_state:
        results = st.session_state.prediction_results
        st.info("Previous prediction results:")
        st.write(f"**Predicted Digit:** {results['digit']}")
        st.write(f"**Confidence:** {results['confidence']:.1f}%")
    else:
        st.info("👆 Click 'Predict Digit' to see results")

# Sidebar with model information
with st.sidebar:
    st.header("ℹ️ Model Information")

    # Model architecture
    with st.expander("🏗️ Model Architecture", expanded=True):
        st.markdown(
            """
        - **Type:** Enhanced CNN with Batch Normalization
        - **Input:** 28×28 grayscale images
        - **Output:** 10 digit probabilities (0-9)
        - **Backend:** TensorFlow/Keras
        - **Inference:** CPU optimized
        """
        )

    # Training details
    with st.expander("📚 Training Details"):
        st.markdown(
            """
        - **Dataset:** MNIST (60k training, 10k test)
        - **Expected Accuracy:** ~99%+
        - **Framework:** TensorFlow/Keras
        - **Epochs:** 30 (with early stopping)
        - **Optimizer:** Adam with learning rate scheduling
        """
        )

    # Tips section
    st.header("💡 Tips for Best Results")
    tips = [
        "**Draw clearly** and fill most of the canvas area",
        "**Center your digit** for better recognition",
        "**Avoid decorations** or extra marks",
        "**Make digits large enough** to be clearly visible",
    ]

    for tip in tips:
        st.markdown(f"• {tip}")

    # Model details section
    st.header("🔧 Technical Details")
    if st.checkbox("Show Model Details"):
        try:
            st.metric("Parameters", f"{model.count_params():,}")
            st.metric("Layers", len(model.layers))
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Output Shape:** {model.output_shape}")
            st.write(f"**Model Type:** {type(model).__name__}")
        except Exception as e:
            st.error(f"Could not load model details: {e}")

# Footer
st.markdown("---")
st.markdown("**Enhanced MNIST Digit Recognizer** - Built with Streamlit & TensorFlow")
