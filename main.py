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
import shutil

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
    # data = await file.read()
    # img_np = load_image_into_numpy_array(data) 
    # text = reader.readtext(img_np)

    IMGSDIR: str= "./test_folder/"

    # file.filename = f"{uuid.uuid4()}.jpg"
    # contents = await file.read()

    # with open(f"{IMGSDIR}{file.filename}", 'wb') as f:
    #     f.write(contents)

    files = os.listdir(IMGSDIR)
    temp_path = f"{IMGSDIR}{files[0]}"
    text = reader.readtext(temp_path)

    return {'data': str(text)}

@app.post('/ocr_images')
async def get_text_from_folder(IMGSDIR: str= "./test_folder/"):
    try:        
        files = os.listdir(IMGSDIR)
        data = {}
        print(files)

        for file in files:
            temp_path = f"{IMGSDIR}{file}"

            # !!!,",..??"
            if file.split('.')[-1] in ['PNG', 'png', 'JPG']:
                continue
                # 'test_folder/file_name.png'

                im = Image.open(temp_path)
                temp_path = temp_path.split('.')[0]+'.jpg'
                im.save(temp_path)
            
            row_text = reader.readtext(temp_path)
            text=''
            for place, t, score in row_text:
                text+=t+' '
                
            data[file] = text


        # df = pd.DataFrame(data)
        # df.to_csv('./test.csv')

        return {'status': 'ok?', 'data': data}
    except:
        return {'status': 'error'}
    

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
