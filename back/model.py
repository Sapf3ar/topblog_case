import numpy as np
import easyocr
from typing import List
import onnxruntime as ort
import cv2
import logging
from collections import Counter
import os
import pandas as pd
import sqlite3


class Model:
    def __init__(self, onnx_path):
        self.PageClassifier =ort.InferenceSession(onnx_path)
        self.cls_values = {}
        self.Reader = easyocr.Reader(['ru', 'en'])

    def __call__(self, imagesPath):
        for imPath in imagesPath:
            im = self.process_image(imPath)
            cls_value = self.get_classifier_prediction(im)
            self.cls_values[imPath] = cls_value
        self.getWrongClasses()
        #self.getEmptyPreds()

    def create_db():
        df = load_broken_data()
        conn = sqlite3.connect('temp/database.db')
        cursor = conn.cursor()

        if not(is_data_exist()):

            create_table_query = '''
            CREATE TABLE IF NOT EXISTS broken_table (
                incorrect_images TEXT,
                reason TEXT
            )
            '''
            cursor.execute(create_table_query)

            for index, row in df.iterrows():
                cursor.execute("INSERT INTO broken_table  (incorrect_images, reason) VALUES (?, ?)", (row['Некорректные изображения'], row['Причина']))

            select_query = '''
            SELECT * FROM broken_table
            '''
            df = pd.read_sql_query(select_query, conn)
            conn.commit()
            conn.close()
            data_is_here()


    def get_preds(self, keywords : List, paths : List, reader, platform : str, n_clusters : int = 10):
        self.predicts = {}
        for path in paths:
            result = Reader.readtext(path, min_size=4, low_text=0.1, link_threshold=0.1, add_margin=0.2)
            boxes = [np.array(item[0]) for item in result]
            texts = [np.array(item[1]) for item in result]
            clustered_boxes, clustered_texts = agglomerative_cluster_boxes(boxes, texts, n_clusters)
            if platform == 'vk':
                pass
            if platform == 'tg':
                pass
            if platform == 'yt':
                pass
            if platform == 'zn':
                pass
        return self.predicts

    def getEmptyPreds(self):
        emptyPreds = []
        for k, v in self.predicts.items():
            if v == 0:
                emptyPreds.append(k)
        reason = "На изображении отсутствует целевая метрика"
        self.writeBroken(reason=reason,
                    images=emptyPreds)

    def process_image(self, image_path: str):
        logging.warning(f"path to image {image_path}")
        img_np = cv2.imread(image_path)
        return img_np

    def getWrongClasses(self):
        invalidImages = []
        classesCounter= Counter(list(self.cls_values.values())).most_common()
        mainClass = classesCounter[0][0]
        for imPath, clsValue in self.cls_values.items():
            if clsValue != mainClass:
                invalidImages.append(imPath)
        reason = "Изображение из другой соц. сети"
        self.writeBroken(reason, invalidImages)

    def writeBroken(self, reason:str,images ):
        dfInvalidImages = {"Некорректные изображения":images,
                           "Причина": [reason]*len(images)}
        if os.path.isfile("temp/broken.csv"):
            df1 = pd.read_csv("temp/broken.csv")
            out = pd.concat([df1, pd.DataFrame(dfInvalidImages)])
            out.to_csv("temp/broken.csv", index=False)
        else:
            pd.DataFrame(dfInvalidImages).to_csv("temp/broken.csv", index=False)

    def inferOCR(self, im):
        pass
    
    def buildGraph(self, imData):
        pass
    
    def drawBoxes(self, im, singleImData):
        '''return the same image for a while'''
        
        # ~~~
        graphs_image = im

        return graphs_image
    

    def toTable(self, dataInferred):
        pass
    
    def get_classifier_prediction(self, image):
        im_ = cv2.resize(image, (512, 512), cv2.INTER_CUBIC)
        
        im_ = np.expand_dims(im_.T, 0).astype(np.float32)
        im_ /= 255
        output = self.PageClassifier.run(None, {"input":im_})
        return label_to_class(output[0].argmax(1))

def label_to_class(label):
    if label == 0:
        return "tg"
    if label == 1:
        return "vk"
    if label == 2:
        return "yt"
    if label == 3:
        return "dz"
    return "error"

def load_broken_data():
    '''
    return pd.DataFrame from temp/broken.csv
    '''
    df = pd.read_csv("temp/broken.csv")
    return df

def is_data_exist():
    '''
    Check text file vars.txt: is database already full
    '''
    with open('vars.txt', 'r') as f:
        a = eval(f.readline())

    if a['is_data_exist']==0:
        return False
    else:
        return True
    
def data_is_here():
    '''
    rewrite text file vars.txt to data is exist
    '''
    with open('vars.txt', 'w') as f:
        f.write("{'is_data_exist': 1}")






