import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️")

st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and get an automatic description of what's in it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Generating caption...'):
        # Load BLIP model and processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Prepare inputs
        inputs = processor(image, return_tensors="pt")

        # Generate caption
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.success("✅ Caption Generated!")
    st.markdown(f"### 📝 Caption: **{caption}**")

st.markdown("---")
st.markdown("Made with ❤️ using BLIP and Streamlit.")
