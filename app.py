import os
import streamlit as st
from src.embed import load_model, embed_images_batched
from src.search import search

st.set_page_config(page_title="Visual Search", layout="wide")
st.title("🔍 Visual Search Engine")
st.caption("Search images using plain English — powered by CLIP")

COCO_DIR = r"C:\Users\prith\fiftyone\coco-2017\validation\data"

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_embeddings():
    model, preprocess, tokenizer = get_model()
    image_paths = [os.path.join(COCO_DIR, f) for f in os.listdir(COCO_DIR)]
    image_vectors, valid_paths = embed_images_batched(image_paths, model, preprocess)
    return image_vectors, valid_paths

with st.spinner("Loading model and embedding images (first run only)..."):
    model, preprocess, tokenizer = get_model()
    image_vectors, valid_paths = get_embeddings()

query = st.text_input("Search your images:", placeholder="a dog running on a beach")

if query:
    results = search(query, model, tokenizer, image_vectors, valid_paths, top_k=6)
    cols = st.columns(3)
    for i, (path, score) in enumerate(results):
        with cols[i % 3]:
            st.image(path, caption=f"score: {score:.3f}", use_container_width=True)