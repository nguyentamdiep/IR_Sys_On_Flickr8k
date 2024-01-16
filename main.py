import streamlit as st
import torch
import clip
from PIL import Image
import os
import pandas as pd
import numpy as np
import faiss
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_path = "flickr8k/Images/"
image_df = pd.read_csv("image_df.csv")


def text_embedding_use_clip(text):
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([text]).to(device)
    text_feature = model.encode_text(text)
    text_feature = text_feature.detach().cpu().numpy()
    return text_feature[0]


def image_embedding_use_resnet50(img):
    base_model=tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(512,512,3)

    )

    model=Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
    ])

    img = cv2.resize(img, (512, 512))
    a = []
    a.append(img)
    a = np.array(a)
    pred = model.predict(a)
    return pred[0]

top_k = st.text_input("Nhập số lượng kết quả muốn trả về", "100")
top_k = int(top_k)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
on_click_search_image = st.button("search image")

query = st.text_input('enter query', '')
on_click_search_text_query = st.button("search text query")

if on_click_search_image:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    #img_resize = cv2.resize(img, (250, 250))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption = "uploaded image", width=200)
    st.write("các ảnh tương đồng: ")
    with st.spinner('Wait for it...'):
        xb = np.load("feature_vectors_image_resnet50.npy")
        index = faiss.IndexFlatL2(2048)
        index.add(xb)
        xq = []
        xq.append(image_embedding_use_resnet50(img))
        xq = np.array(xq)
        D, I = index.search(xq, top_k)
        result = []
        for i in range(len(I[0])):
            result.append(image_df["file_name"][I[0][i]])
        list_result_image_path = []
        list_captions = []
        for file_name in result:
            list_result_image_path.append(dataset_path + file_name)
            list_captions.append(dataset_path + file_name)
        
    st.image(list_result_image_path, caption = list_captions, width=200)


if on_click_search_text_query:
    if query != "":
        st.write("Kết quả:")
        with st.spinner('Wait for it...'):
            xb = np.load("feature_vectors_image_clip_vit_b32.npy")
            index = faiss.IndexFlatL2(512)
            index.add(xb)
            xq = []
            xq.append(text_embedding_use_clip(query))
            xq = np.array(xq)
            D, I = index.search(xq, top_k)
            result = []
            for i in range(len(I[0])):
                result.append(image_df["file_name"][I[0][i]])
        
        list_result_image_path = []
        list_captions = []
        for file_name in result:
            list_result_image_path.append(dataset_path + file_name)
            list_captions.append(dataset_path + file_name)
        
        st.image(list_result_image_path, caption = list_captions, width=200)






