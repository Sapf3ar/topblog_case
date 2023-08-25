from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
# from pydantic import BaseModel

import io
# import torch
import numpy as np
import zipfile

from PIL import Image

import os
from fastapi import FastAPI, File, UploadFile, status
from fastapi.exceptions import HTTPException
import aiofiles

CHUNK_SIZE = 1024 * 1024 

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# import tensorflow as tf
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils

app = FastAPI(title='Test')

# detect_fn = tf.saved_model.load('saved_model')
# category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt",use_display_name=True)


def load_image_into_numpy_array(data):
    return np.array(Image.open(io.BytesIO(data)))


def predict(image):
     # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    image_np = load_image_into_numpy_array(image)

    return image_np



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

        directory_to_extract = './data'

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


@app.post("/image")
async def create_upload_file(image: UploadFile = File(...)):
    img_data = await image.read()
    img_np = load_image_into_numpy_array(img_data)
    #print(predicted_image.shape)
    #predicted_image.save("Predicted.jpg")
    # return predicted_image
    return {'data': f'the first pixel is {str(img_np[0][0])}'}
