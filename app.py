import streamlit as st
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
from PIL import Image

# Page Setup
st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️")
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image, choose a feature, and explore captions, styles & translations!")

feature = st.sidebar.selectbox(
    "Choose a feature:",
    ["Multiple Captions", "Caption Styles", "Translate Caption"]
)

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Please upload an image to proceed.")
    st.stop()

def load_image(file):
    return Image.open(file).convert('RGB')

image = load_image(uploaded_file)
st.image(image, caption='Uploaded Image', use_container_width=True)

# Load Models
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") # to prepare image for model to process n decode model output
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

@st.cache_resource
def load_llm():
    model_name = "MBZUAI/LaMini-Flan-T5-783M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_translator(lang):
    mapping = {
        "Hindi": "Helsinki-NLP/opus-mt-en-hi",
        "Spanish": "Helsinki-NLP/opus-mt-en-es",
    }
    return pipeline("translation", model=mapping[lang])

processor, blip_model = load_blip()
llm = load_llm()

# Helper Functions

def generate_caption(): # Use sampling for diverse output
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = blip_model.generate(
        **inputs, #unpack input dictionary
        max_new_tokens=50,
        #do_sample=True,
        top_k=40
    )
    caption = processor.decode(outputs[0], skip_special_tokens=True) #removes padding tokens
    return caption


def styled_caption(caption, style):
    prompts = {
        "Descriptive": f"Make this detailed and descriptive: {caption}",
        "Funny": f"Make a humorous, witty caption from this: {caption}",
        "Poetic": f"Transform this into a poetic expression: {caption}"
    }
    response = llm(
        prompts[style],
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7
    )
    return response[0]['generated_text']

# Features

# 1. Multiple Captions
def do_multiple_captions():
    n = st.slider("Number of captions", 1, 5, 3)
    strategy = st.selectbox("Strategy", ["Beam Search", "Sampling"])

    gen_kwargs = {"max_new_tokens": 50} #to add parameters while calling the model
    if strategy == "Beam Search":
        gen_kwargs.update({"num_beams": 5, "num_return_sequences": n})#explore 5 possible sequences and return n of those
    else:
        gen_kwargs.update({"do_sample": True, "top_k": 50, "num_return_sequences": n})

    inputs = processor(image, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs, **gen_kwargs)
    captions = [processor.decode(o, skip_special_tokens=True) for o in outputs]

    st.success("Generated Captions:")
    for idx, cap in enumerate(captions, 1):
        st.write(f"**{idx}.** {cap}")

# 2. Styled Captions
def do_caption_styles():
    base_caption = generate_caption()
    style = st.selectbox("Choose style", ["Descriptive", "Funny", "Poetic"])
    styled = styled_caption(base_caption, style)
    st.success(f"**{style} Caption:** {styled}")

# 3. Translate Caption
def do_translate():
    base_caption = generate_caption()
    lang = st.selectbox("Translate to", ["Hindi", "Spanish"])
    translator = load_translator(lang)
    result = translator(base_caption)
    translated = result[0]['translation_text']
    st.success(f"Original: {base_caption}")
    st.success(f"Translated ({lang}): {translated}")

# Feature setup
with st.spinner('Processing...'):
    if feature == "Multiple Captions":
        do_multiple_captions()
    elif feature == "Caption Styles":
        do_caption_styles()
    elif feature == "Translate Caption":
        do_translate()

st.markdown("---")
st.markdown("Made with ❤️ using BLIP, LaMini‑Flan‑T5‑783M, HuggingFace and Streamlit.")