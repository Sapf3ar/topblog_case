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


st.write("Поиск неправильных изображений")

@st.cache_data
def load_broken_data():
    df = pd.read_csv("temp/broken.csv")
    return df

@st.cache_data
def is_data_exist():
    with open('vars.txt', 'r') as f:
        a = eval(f.readline())

    if a['is_data_exist']==0:
        return False
    else:
        return True
    
def data_is_here():
    with open('vars.txt', 'w') as f:
        f.write("{'is_data_exist': 1}")


df = load_broken_data()

img_names =list(df[df.columns[0]])

def show_img(path: str):
    st.image(path)

def find_similar(temptext: str = 't'):
    top = {
        '1': ['',100],
        '2': ['',100],
        '3': ['',100]
           }
    for filename in img_names:
        dist_lev = lev(filename, temptext)
        dist1 = 0 if temptext in filename else dist_lev + 1
        dist = min(dist_lev, dist1)

        # bubble
        if dist < top['3'][1] and temptext!= 't':
            top['3'][0] = filename
            top['3'][1] = dist

            if dist < top['2'][1]:
                top['3'][0] = top['2'][0]
                top['3'][1] = top['2'][1]

                top['2'][0] = filename
                top['2'][1] = dist

                if dist < top['1'][1]:
                    
                    top['2'][0] = top['1'][0]
                    top['2'][1] = top['1'][1]

                    top['1'][0] = filename
                    top['1'][1] = dist
            
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