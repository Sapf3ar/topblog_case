import requests
import os
from requests_toolbelt import MultipartEncoder
import streamlit as st
import logging

st.set_page_config(
    page_title="Загрузка данных",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
    }
)
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)




# interact with FastAPI endpoint

backend_upload = "http://127.0.0.1:8000/upload"
backend_download = "http://127.0.0.1:8000/download"
#ibackend_model = "http://127.0.0.1:8000/create_model"

#backend_model_exists = "http://127.0.0.1:8000/model_exists"

def process(image, server_url: str, filename:str=""):

    m = MultipartEncoder(fields={"file": (filename, image, "image")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r




col1, col2 = st.columns(2)
autoSocialNetwork = col1.checkbox("Автоматическое распознавание соц-сети")
if not autoSocialNetwork: 
    task_type = col1.selectbox(
            'Выберите Соц. Сеть:',
        ('Телеграм', 'ВКонтакте', 'YouTube', 'Дзен'), 
        placeholder = " ")

    col1.write(f'Вы Выбрали: {task_type}')
files = st.file_uploader("insert image", 
                           accept_multiple_files=True,
                           )
def createDownloadButton():

    file = requests.get(backend_download)
    logging.warning(os.listdir("."))
    fileName = file.headers['content-disposition'].split(";")[-1].split("=")[-1].strip('"./')
    with open(fileName, 'rb') as f:
        st.download_button("download excel table", data = f, file_name=fileName.strip("./"))

if st.button("Отправить данные:"):
    if files is not None:    
        for file in files:
            bytes_data = file.getvalue()
            name = file.name
            segments = process(bytes_data, backend_upload, name)
    if segments.ok:
        createDownloadButton()
# interact with FastAPI endpoint

#ibackend_model = "http://127.0.0.1:8000/create_model"

#backend_model_exists = "http://127.0.0.1:8000/model_exists"





