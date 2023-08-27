
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



# st.write("Поиск неправильных изображений")

# @st.cache_data
# def load_broken_data():
#     df = pd.read_csv("./broken.csv")
#     return df

# @st.cache_data
# def is_data_exist():
#     with open('vars.txt', 'r') as f:
#         a = eval(f.readline())

#     if a['is_data_exist']==0:
#         return False
#     else:
#         return True
    
# def data_is_here():
#     with open('vars.txt', 'w') as f:
#         f.write("{'is_data_exist': 1}")


# conn = sqlite3.connect('./database.db')
# cursor = conn.cursor()

# if not(is_data_exist()):
    
#     df = load_broken_data()

#     create_table_query = '''
#     CREATE TABLE IF NOT EXISTS broken_table (
#         incorrect_images TEXT,
#         reason TEXT
#     )
#     '''
#     cursor.execute(create_table_query)

#     for index, row in df.iterrows():
#         cursor.execute("INSERT INTO broken_table  (incorrect_images, reason) VALUES (?, ?)", (row['Некорректные изображения'], row['Причина']))

    
#     conn.commit()
#     conn.close()
#     data_is_here()


# select_query = '''
#     SELECT * FROM broken_table
#     '''
# df = pd.read_sql_query(select_query, conn)
    
# st.write('Loaded data:')
# st.write(df)

# img_names =list(df[df.columns[0]])

# def show_img(path: str):
#     st.image(path)

# def find_similar(temptext: str = 't'):
#     top = {
#         '1': ['',100],
#         '2': ['',100],
#         '3': ['',100]
#            }
#     for filename in img_names:
#         dist_lev = lev(filename, temptext)
#         dist1 = 0 if temptext in filename else dist_lev + 1
#         dist = min(dist_lev, dist1)

#         # bubble
#         if dist < top['3'][1] and temptext!= 't':
#             top['3'][0] = filename
#             top['3'][1] = dist

#             if dist < top['2'][1]:
#                 top['3'][0] = top['2'][0]
#                 top['3'][1] = top['2'][1]

#                 top['2'][0] = filename
#                 top['2'][1] = dist

#                 if dist < top['1'][1]:
                    
#                     top['2'][0] = top['1'][0]
#                     top['2'][1] = top['1'][1]

#                     top['1'][0] = filename
#                     top['1'][1] = dist
            
#     b1 = st.button(top['1'][0], type="primary",key='b1')
#     b2 = st.button(top['2'][0], type="primary",key='b2')
#     b3 = st.button(top['3'][0], type="primary",key='b3')
    
#     if b1:
#         show_img(top['1'][0])
#     if b2:
#         show_img(top['2'][0])
#     if b3:
#         show_img(top['3'][0])


# text =  st.text_input('Input')
# logging.warning(text)

# if len(text) > 2:
#     find_similar(text)



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
