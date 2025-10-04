import streamlit as st
from PIL import Image
from inference import predict  # your model class





# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🍛 Indian Food Classifier")

st.write("Upload an image of food or take a picture to identify the dish.")

# Upload or take a picture
img_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
camera_file = st.camera_input("Or take a picture")

# Prioritize camera if both given
if camera_file is not None:
    img_file = camera_file

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Dish"):
        label = predict(image)
        st.success(f"Predicted Dish: **{label}**")

