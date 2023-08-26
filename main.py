# from pydantic import BaseModel
import sqlite3

import io
# import torch
import numpy as np
import zipfile

import sqlite3
import pandas as pd

from PIL import Image
import PIL


import os
from fastapi import FastAPI, File, UploadFile, status
from fastapi.exceptions import HTTPException
import aiofiles

import easyocr

CHUNK_SIZE = 1024 * 1024 
reader = easyocr.Reader(['ru'])

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
app = FastAPI(title='SP-topblog')


@app.post('/')
async def main():
    return {'data': None}


@app.post("/upload")
async def upload_zip(file: UploadFile = File(...)):
    global images_list

    try:
        filepath = os.path.join('./', os.path.basename(file.filename))
        async with aiofiles.open(filepath, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                await f.write(chunk)
        await file.close()

        directory_to_extract = './'

        for address, dirs, files in os.walk('./'):
            for name in files:
                path_to_zip_file = "./"

                if name.split('.')[-1] == 'zip':
                    
                    directory = name.split('.')[0]
                    path_to_zip_file += name

                    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                        zip_ref.extractall(directory_to_extract)
            break
        
        images_list=[]
        
        for address, dirs, files in os.walk(f'./{directory}'):
            images_list = files
            break
        
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail='There was an error uploading the file')
    finally:
        pass
    return {"message": f"Successfuly uploaded {file.filename} to {directory}, list's len = {images_list}"}

@app.get('/check')
async def check_folders():
    info = {'msg': []}
    for address, dirs, files in os.walk('./'):
        info['msg'].append(dirs)

    return info

@app.post("/image")
async def create_upload_file(image: UploadFile = File(...)):
    img_data = await image.read()
    img_np = load_image_into_numpy_array(img_data)
    #print(predicted_image.shape)
    #predicted_image.save("Predicted.jpg")
    # return predicted_image
    return {'data': f'the first pixel is {str(img_np[0][0])}'}


@app.post('/ocr_image')
async def get_text_from_image(file: UploadFile = File(...)):
    '''
    Не удается нормально сделать
    '''
    data = await file.read()
    img_np = load_image_into_numpy_array(data) 
    text = reader.readtext(img_np)

    # IMGSDIR = "./test_images/"

    # file.filename = f"{uuid.uuid4()}.jpg"
    # contents = await file.read()

    # with open(f"{IMGSDIR}{file.filename}", 'wb') as f:
    #     f.write(contents)

    # files = os.listdir(IMGSDIR)
    # temp_path = f"{IMGSDIR}{files[0]}"

    return {'img': file.filename, 'data': text}


@app.post('/sql')
def pandas_sqlite():
    con = sqlite3.connect('test.db') # test.db is exist
    cur = con.cursor()

    df = pd.DataFrame({
        'index': [1,5,6,2],
        'metric': [23,4000,982,11]
        })
    
    df.to_sql('scrinshots', con, index=False)
    cur.execute("SELECT * FROM scrinshots")



def load_image_into_numpy_array(data):
    return np.array(Image.open(io.BytesIO(data)))
