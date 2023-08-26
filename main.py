import os
from fastapi import FastAPI, File, UploadFile, status
from fastapi import Response
from fastapi.exceptions import HTTPException
import aiofiles
import sqlite3
import io

import numpy as np
import zipfile

import sqlite3
import pandas as pd

from PIL import Image
import PIL
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
    '''
    Загрузка зип файла с изображениями
    '''
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
    '''
    Папки в корневой
    '''
    info = {'msg': []}
    for address, dirs, files in os.walk('./'):
        info['msg'].append(dirs)
    return info


@app.post('/ocr_images')
async def get_text_from_folder(IMGSDIR: str= "./test_folder/"):
    try:        
        files = os.listdir(IMGSDIR)
        data = {'img':[], 'text':[]}
        data['img'] = files

        print(files)
        for file in files:
            temp_path = f"{IMGSDIR}{file}"
            
            text = '  '.join(reader.readtext(temp_path,  detail=0))
            print(text)
            data['text'].append(text)


        # DB надо дебажить
        # df = pd.DataFrame.from_dict(data)
        # pandas_to_sql(df)

        return {'status': 'ok', 'data': data}
    except:
        return {'status': 'error'}


def pandas_to_sql(df):
    con = sqlite3.connect('test.db') # empty 'test.db' is exist
    cur = con.cursor()
    df.to_sql('scrinshots', con, index=False)
    cur.execute("SELECT * FROM scrinshots")

    
def load_image_into_numpy_array(data):
    return np.array(Image.open(io.BytesIO(data)))



# @app.post("/image")
async def create_upload_file(image: UploadFile = File(...)):
    '''
    Не используется
    '''
    img_data = await image.read()
    img_np = load_image_into_numpy_array(img_data)
    #print(predicted_image.shape)
    #predicted_image.save("Predicted.jpg")
    # return predicted_image
    return {'data': f'the first pixel is {str(img_np[0][0])}'}


# @app.post('/ocr_image')
async def get_text_from_image(file: UploadFile = File(...)):
    '''
    Не дебажилось и не используется
    '''
    data = await file.read()
    img_np = load_image_into_numpy_array(data) 
    text = '  '.join(reader.readtext(img_np,  detail=0))
    
    return {'data': str(text)}
