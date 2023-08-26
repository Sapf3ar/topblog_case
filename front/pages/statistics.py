
import io
from PIL import Image
import requests
import numpy as np
import logging
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st


url_backend_draw = "http://127.0.0.1:8000/draw"

def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)

    return r

input_image = st.file_uploader("Insert image", accept_multiple_files=False)

if st.button("Get result"):
    col1, col2 = st.columns(2)
    if input_image is not None:
        logging.warning("Click")

        bytes_data = input_image.getvalue()
        segments = process(bytes_data, url_backend_draw)

        original_image = np.asarray(Image.open(io.BytesIO(bytes_data)).convert("RGB"))
        graphs_image =np.asarray(Image.open(io.BytesIO(segments.content)).convert("RGB"))

        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Graph")
        col2.image(graphs_image, use_column_width=True)