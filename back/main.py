from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import HTTPException
from typing import List, Literal, Any, Tuple
import logging
import aiofiles
import os
import zipfile
from PIL import Image
import io
import cv2
import glob
import numpy as np
from back.yadisk import downloadYaDisk
import pandas as pd
CHUNK_SIZE = 1024 * 1024  # adjust the chunk size as desired
app = FastAPI()

def getAllowedExtensions() -> List[str]:
    return ["jpg", "jpeg", "png", "bmp"]

def infer_model() -> pd.DataFrame:
    pass

async def writeFile(filePath:str, file:UploadFile) -> str:
    async with aiofiles.open(filePath, 'wb') as f:
        while chunk := await file.read(CHUNK_SIZE):
            await f.write(chunk)
    await file.close()
    return filePath

@app.get("/download")
async def send_csv_table() -> FileResponse:
    infer_model()
    downloadPath = "./test.xlsx"
    dummy_csv = pd.DataFrame({"test":['1', '2'], 
                              "train":['3', '4']})
    dummy_csv.to_excel(downloadPath)
    return FileResponse(downloadPath, filename = downloadPath)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    '''
    Загрузка зип файла с изображениями
    '''
    out_dir = "./out_data"
    if "zip" not in file.filename:
        filepath = os.path.join(out_dir, os.path.basename(file.filename))
    else:
        filepath = os.path.join('./', os.path.basename(file.filename))

    logging.warning(file.filename)
    await writeFile(filepath,file) 
    if "zip" in filepath:
        unzip_file(filepath, out_dir)
        
    validImages, invalidImages = getImagesPath(out_dir)
    return JSONResponse({"validImages":validImages, 
                  "invalidImages":invalidImages})
        
        
def getImagesPath(images_path:str) -> Tuple[List[str], List[str]]:
    allowed_ext = getAllowedExtensions()
    validFiles = []
    invalidFiles = []
    for filePath in os.listdir(images_path):
        fullPath = os.path.join(images_path, filePath)
        if filePath.split(".")[-1] not in allowed_ext:
            invalidFiles.append(fullPath)
        else:
            validFiles.append(fullPath)
    return (validFiles, invalidFiles)
        
def unzip_file(file_path:str, out_dir:str) -> str:
    logging.warning(f"in unzip func {file_path}")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

