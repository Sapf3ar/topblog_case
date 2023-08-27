import requests
import os
from requests_toolbelt import MultipartEncoder
import streamlit as st
import logging
from pydantic import BaseModel


class DiskUrl(BaseModel):
    url:str

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


backend_upload = "http://127.0.0.1:8000/upload"
backend_download = "http://127.0.0.1:8000/download"
backend_upload_yandex = "http://127.0.0.1:8000/upload_yandex"


def process(image, server_url: str, filename:str="", task_type:str=''):

    m = MultipartEncoder(fields={"file": (filename, image, "image")})
    if not task_type:
        task_type = "automatic"
    r = requests.post(
            server_url, data=m, headers={"Content-Type": m.content_type, "task-type":"auto"}, json={"task_type":task_type}, timeout=8000
    )

    return r

def createDownloadButton():
    file = requests.get(backend_download)
    fileName = file.headers['content-disposition'].split(";")[-1].split("=")[-1].strip('"./')
    with open(fileName, 'rb') as f:
        st.download_button("download excel table", data = f, file_name=fileName.strip("./"))


col1, col2 = st.columns(2)
autoSocialNetwork = col1.checkbox("Автоматическое распознавание соц-сети")
if not autoSocialNetwork: 
    task_type = col1.selectbox(
            'Выберите Соц. Сеть:',
        ('Телеграм', 'ВКонтакте', 'YouTube', 'Дзен'), 
        placeholder = " ")

    col1.write(f'Вы Выбрали: {task_type}')


url = st.text_input("Введите ссылку на Яндекс.Диск:")
task_type = task_type if task_type else "auto"
if st.button("Скачать данные с Яндекс.Диск"):
    ans = requests.post(backend_upload_yandex, json={"url":url, "task_type":task_type})


files = st.file_uploader("insert data", 
                           accept_multiple_files=True,
                           )
if st.button("Отправить данные:"):
    if files is not None:    
        for file in files:
            bytes_data = file.getvalue()
            name = file.name
            segments = process(bytes_data, backend_upload, name, task_type=task_type)
            
    if segments.ok or ans.ok:

        col1, col2 = st.columns(2)
        with col1:            
            createDownloadButton()
        with col2:
            with open("temp/broken.csv", 'rb') as f:

                st.download_button("Скачать список некорректных изображения",\
                        data = f, file_name="temp/broken.csv")

            
# interact with FastAPI endpoint

#ibackend_model = "http://127.0.0.1:8000/create_model"

#backend_model_exists = "http://127.0.0.1:8000/model_exists"





