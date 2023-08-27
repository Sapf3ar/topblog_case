
import io
from PIL import Image
import requests
import numpy as np
import logging
from requests_toolbelt.multipart.encoder import MultipartEncoder

from Levenshtein import distance as lev
import pandas as pd
import logging
import sqlite3

import streamlit as st



url_backend_draw = "http://127.0.0.1:8000/draw"

def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
    return r

st.write("Построение графов")
input_image = st.file_uploader("Insert image", accept_multiple_files=False)



if st.button("Get result"):
    col1, col2 = st.columns(2)
    if input_image is not None:
        logging.warning("Click")

        bytes_data = input_image.getvalue()
        procces_data = process(bytes_data, url_backend_draw)

        original_image = Image.open(io.BytesIO(bytes_data))
        graphs_image = Image.open(io.BytesIO(procces_data.content))

        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Graph")
        col2.image(graphs_image, use_column_width=True)


# 




#sql


# col1, col2 = st.columns(2)
# task_type = col1.selectbox(
#         'What is metric',
#     ('vr', 'arr', 'sub')
# )
# col1.write(f'Вы Выбрали: {task_type}')


# filtered_df = df[df['metric_name']== f'{task_type}']
# st.write(filtered_df)
# value = float(filtered_df['metric_value'].mean())
# st.write('Mean value is ', value)


# filtered_df = filtered_df[filtered_df['metric_value'] < value]
# result = filtered_df['user_id']

# x = st.slider('Table of users with less value(head):', value=10)


# # '''
# # sql_request = SELECT user_id 
# # FROM images_table 
# # WHERE metric_name = 'task_type' 
# #   AND metric_value < (
# #       SELECT AVG(metric_value) 
# #       FROM images_table 
# #       WHERE metric_name = 'task_type'
# #   )
# # '''

# st.write(result.head(x))
# st.write('Graphs interpreter')

# st.image('./front/test.jpg')
