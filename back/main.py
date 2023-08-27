from fastapi import FastAPI, File, UploadFile, status, Request
import glob
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.exceptions import HTTPException
from typing import List, Literal, Any, Tuple
import logging
import aiofiles
import os
import zipfile
import cv2
from pydantic import BaseModel
from back.yadisk import downloadYaDisk
import pandas as pd
import sqlite3
from back.model import Model
import shutil
con = sqlite3.connect("tutorial.db")
CHUNK_SIZE = 1024 * 1024  # adjust the chunk size as desired


app = FastAPI()

class DiskUrl(BaseModel):
    url:str

def getAllowedExtensions() -> List[str]:
    return ["jpg", "jpeg", "png", "bmp", "PNG", "JPG"]

def infer_model(imPaths:List[str]) -> pd.DataFrame:
    model = Model("../dataset/classifier.onnx")
    outs = model(imPaths)
    return outs

async def writeFile(filePath:str, file:UploadFile) -> str:
    async with aiofiles.open(filePath, 'wb') as f:
        while chunk := await file.read(CHUNK_SIZE):
            await f.write(chunk)
    await file.close()
    return filePath

@app.get("/download")
async def send_csv_table() -> FileResponse:

    downloadPath = "temp/data.xlsx"
    return FileResponse(downloadPath, filename = downloadPath)

@app.post("/upload")
async def upload(file: UploadFile = File(task_type="task_type")):
    '''
    Загрузка зип файла с изображениями
    '''
    logging.warning(file)
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
    reason = "Некорректный формат изображения"
    dfInvalidImages = {"Некорректные изображения":invalidImages,
                       "Причина": [reason]*len(invalidImages)}
    pd.DataFrame(dfInvalidImages).to_csv("temp/broken.csv", index=False)
    outs = infer_model(validImages)
    pd.DataFrame(outs).to_excel("temp/data.xlsx", index=False)

    #cleanOutDir(out_dir)
    return JSONResponse({"validImages":validImages, 
                  "invalidImages":invalidImages})

def cleanOutDir(out_dir:str):
    files = glob.glob(out_dir+"/*")
    for filePath in files:
        os.system(f"rm -rf {filePath}")

def getInputDir(out_dir:str):
    imagesPaths = []
    for  root, subdirs, files in os.walk(out_dir):
        for f in files:
            imagesPaths.append(os.path.join(root, f))
        for subdir in subdirs:
            for filePath in os.listdir(os.path.join(root, subdir)):
                if not os.path.isdir(filePath):
                    imagesPaths.append(os.path.join(root, subdir, filePath))

    return imagesPaths
@app.post('/draw')
async def draw_image(url:DiskUrl):
    model = Model("../dataset/classifier.onnx")
    im = cv2.imread(os.path.join("out_data", url.url))
    im_ = model.drawBoxes(im)
    imb = cv2.imencode(".png", im_)[1].tobytes()
    return Response(imb, media_type="image/png")


@app.post("/upload_yandex")
def load_yandex(url:DiskUrl):
    out_dir = "./out_data"
    logging.warning(url)
    filePath = downloadYaDisk(url.url)
    unzip_file(filePath, "./out_data")

    validImages, invalidImages = getImagesPath(out_dir)
    reason = "Некорректный формат изображения"
    dfInvalidImages = {"Некорректные изображения":invalidImages,
                       "Причина": [reason]*len(invalidImages)}
    pd.DataFrame(dfInvalidImages).to_csv("temp/broken.csv", index=False)
    outs = infer_model(validImages)
    cleanOutDir(out_dir)
    pd.DataFrame(outs).to_excel("temp/data.xlsx", index=False)
    return filePath

def getImagesPath(images_path:str) -> Tuple[List[str], List[str]]:
    allowed_ext = getAllowedExtensions()
    imagesPathList = getInputDir(images_path)
    validFiles = []
    invalidFiles = []
    for filePath in imagesPathList:
        if filePath.split(".")[-1] not in allowed_ext:
            invalidFiles.append(filePath)
        else:
            validFiles.append(filePath)
    return (validFiles, invalidFiles)
        
def unzip_file(file_path:str, out_dir:str) -> str:
    logging.warning(f"in unzip func {file_path}")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
    return file_path
