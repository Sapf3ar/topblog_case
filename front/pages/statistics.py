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
from pydantic import BaseModel
import streamlit as st
import requests

class Url:
    url:str

st.write("Поиск неправильных изображений")

@st.cache_data
def load_broken_data():
    df = pd.read_csv("temp/broken.csv")
    return df


backend_draw= "http://127.0.0.1:8000/draw"

df = load_broken_data()

img_names =list(df[df.columns[0]])

def show_img(path: str):
    
    r = requests.post(
            backend_draw, json={"url":path} )
    im = Image.open(io.BytesIO(r.content))
    st.image(im)

def find_similar(temptext: str = 't'):
    top = {
        '1': ['',100, ''],
        '2': ['',100, ''],
        '3': ['',100, '']
           }
    for filename in img_names:
        dist_lev = lev(filename, temptext)
        dist1 = 0 if temptext in filename else dist_lev + 1
        dist = min(dist_lev, dist1)
        fileNamePruned = filename.split("/")[-1]
        # bubble
        if dist < top['3'][1] and temptext!= 't':
            top['3'][0] = fileNamePruned
            top['3'][1] = dist            
            if dist < top['2'][1]:
                top['3'][0] = top['2'][0]
                top['3'][1] = top['2'][1]

                top['2'][0] = fileNamePruned
                top['2'][1] = dist

                if dist < top['1'][1]:
                    
                    top['2'][0] = top['1'][0]
                    top['2'][1] = top['1'][1]

                    top['1'][0] = fileNamePruned
                    top['1'][1] = dist
    logging.warning(top)            
    b1 = st.button(top['1'][0], type="primary",key='b1')
    b2 = st.button(top['2'][0], type="primary",key='b2')
    b3 = st.button(top['3'][0], type="primary",key='b3')
    
    if b1:
        show_img(top['1'][0])
    if b2:
        show_img(top['2'][0])
    if b3:
        show_img(top['3'][0])


text =  st.text_input('Input')
logging.warning(text)

if len(text) > 2:
    find_similar(text)
