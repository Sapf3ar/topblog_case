import re
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import easyocr
from typing import List
import onnxruntime as ort
import cv2
import logging
from collections import Counter
import os
import pandas as pd


def visualize_clusters_on_image(img: np.ndarray, clustered_boxes: np.ndarray):
    """make plot with clusters and relations

    Args:
        img (np.ndarray):
        clustered_boxes (np.ndarray): list or np.ndarray in shape N, 4, 2 of N boxes

    Returns:
        np.ndarray
    """
    img = img.copy()
    for boxes in clustered_boxes:
        cluster_color = tuple(np.random.randint(0, 255, 3).tolist())

        central_box = boxes[0]
        central_centroid = compute_centroid(central_box)
        for box in boxes:
            centroid = compute_centroid(box)
            
            # Нанесение линии от центра кластера до каждого бокса
            cv2.line(img, tuple(map(int, centroid)), tuple(map(int, central_centroid)), cluster_color, 1, cv2.LINE_AA)

            # Нанесение бокса на изображение
            bbox = box_to_bbox(box)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), cluster_color, 1)
    return img

def box2key(box):
    return ",".join(np.array(box).astype(int).flatten().astype(str))


def compute_centroid(box):
    """Compute centroid for a given box."""
    x_coords = box[:, 0]
    y_coords = box[:, 1]
    centroid = [np.mean(x_coords), np.mean(y_coords)]
    return centroid

def compute_area(bbox):
    """Compute area of a bbox."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# Convert boxes to bounding boxes
def box_to_bbox(box):
    x_coords = box[:, 0]
    y_coords = box[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    return [x_min, y_min, x_max, y_max]

def agglomerative_cluster_boxes(boxes, texts, n_clusters=10, linkage='ward'):
    centroids = np.array([compute_centroid(box) for box in boxes])
    
    # Применение агломеративной кластеризации
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agglomerative.fit_predict(centroids)

    # Группировка боксов по кластерам
    clustered_boxes = {}
    clustered_texts = {}

    for box, text, label in zip(boxes, texts, labels):
        if label not in clustered_boxes:
            clustered_boxes[label] = []
            clustered_texts[label] = []
        clustered_boxes[label].append(box)
        clustered_texts[label].append(text)

    return clustered_boxes, clustered_texts
from fuzzywuzzy import fuzz, process


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))
  
def get_top_n_boxes_texts(boxes, texts, keyword, min_score=30, n=5):
    if len(boxes) == 0:
        return 0, [], []
    best_match, score = process.extractOne(keyword, texts)  # Для одного ключевого слова!
    if score < min_score:
        return 0, [], []
    # Получаем координаты центра для лучшего соответствия
    best_match_idx = texts.index(best_match)
    best_match_center = np.mean(boxes[best_match_idx], axis=0)
    
    # Вычисляем евклидово расстояние от лучшего соответствия до всех других boxes
    distances = [euclidean_distance(best_match_center, np.mean(box, axis=0)) for box in boxes]

    # Находим индексы топ n ближайших boxes
    top_n_indices = np.argsort(distances)[:n]
    
    top_n_boxes = [boxes[i] for i in top_n_indices]
    top_n_texts = [texts[i] for i in top_n_indices]

    done = False
    for t in top_n_texts:
        if any(c.isdigit() for c in t):
            done = True
        if t == 'все' or t == 'статьи' or t == 'посты':
            unused_boxes = [boxes[i] for i in range(len(boxes)) if i not in top_n_indices]
            unused_texts = [texts[i] for i in range(len(boxes)) if i not in top_n_indices]
            return get_top_n_boxes_texts(unused_boxes, unused_texts, keyword=keyword, min_score=min_score, n=n)
    if done:
        return score, top_n_boxes, top_n_texts
    else:
        unused_boxes = [boxes[i] for i in range(len(boxes)) if i not in top_n_indices]
        unused_texts = [texts[i] for i in range(len(boxes)) if i not in top_n_indices]
        
        return get_top_n_boxes_texts(unused_boxes, unused_texts, keyword=keyword, min_score=min_score, n=n)


def zn_metric(texts : List) -> int:
    nums = []
    for word in texts:
        word = re.sub(r'(?<=\d) +(?=\d)', '',  word)
        if "тыс" in word:
            word = re.sub(r"\D", "", word)
            word += "000"
        try:
            nums.append(int(word))
        except:
            continue
    if len(nums) != 0:
        return max(nums)
    else:
        return 0

def yt1_metric(texts : List) -> int:
    nums = []
    for word in texts:
        word = re.sub(r'(?<=\d) +(?=\d)', '',  word)
        try:
            nums.append(int(word))
        except:
            continue
    if len(nums) != 0:
        return max(nums)
    else:
        return 0

def yt2_metric(texts : List) -> int:
    nums = []
    for word in texts:
        word = re.sub(r'(?<=\d) +(?=\d)', '',  word)
        if "тыс" in word:
            word = word.split(',')
            for i in range(len(word)):
                word[i] = re.sub(r"\D", "",  word[i])
            try:
                num = int(word[0]) * 1000 + int(word[1])
                nums.append(num)
            except:
                continue
            
        try:
            nums.append(int(word))
        except:
            continue
    if len(nums) != 0:
        return max(nums)
    else:
        return 0

def tg_metric(texts : List) -> int:
    nums = []
    for word in texts:
        if "%" in word:
            word = word[:word.rfind('%')]
            num = word.split(' ')
            
            for i in range(len(num) -1, -1, -1):
                n = num[i]
                try:
                    nums.append(float(n))
                    break
                except:
                    continue
    new_nums = sorted(nums, reverse=True)
    for num in new_nums:
        if num < 500:
            return num
    return 0

def vk_metric(texts : List) -> int:
    nums = []
    for word in texts:
        word = re.sub(r'(?<=\d) +(?=\d)', '',  word)
        if word.isnumeric():
            nums.append(word)
        elif "," in word:
            word = word.split(',')
            try:
                nums.append(word[0] + word[1])
            except:
                continue
        else:
            s_arr = word.split(' ')
            for s_word in s_arr:
                if s_word.isnumeric():
                    nums.append(s_word)
                elif "K" in s_word:
                    nums.append(word.split()[0] + "000")
    return get_num_pred_vk(nums)
def get_num_pred_vk(nums : List) -> int:
    new_nums = []
    for i, num in enumerate(nums):
        try:
            new_nums.append(int(num))
        except:
            continue
    new_nums = sorted(new_nums, reverse=True)
    for num in new_nums:
        if num < 2000:
            return num
    return 0
class Model:
    def __init__(self, onnx_path):
        self.PageClassifier =ort.InferenceSession(onnx_path)
        self.cls_values = {}
        self.Reader = easyocr.Reader(['ru', 'en'])
        self.keywords = {'vk'  : ["подписчик", "участник", "друзей"],
            'tg'  : ["VR", "vr", "ERR", "err", "V", "v"],
            'yt' : ["Подписчик", "подписчики"],
                         #'yt2' : ["Просмотры", "просмотры", "Просмотр", "просмотров"],
            'zn'  : ["дочитывания", "просмотры"]}
        self.predicts={}
        self.out_dict = {'vk' : 'Подписчиков',
           'tg' : 'VR',
           'yt' : 'Подписчиков',
           'zn' : 'Просмотры'}

    def __call__(self, imagesPath):
        ims = []
        values = []
        for imPath in imagesPath:
            im = self.process_image(imPath)
            cls_value = self.get_classifier_prediction(im)
            self.cls_values[imPath] = cls_value
            preds = self.get_preds(keywords = self.keywords, 
                           image=im,
                           path= imPath ,
                           reader = self.Reader, 
                           platform = cls_value)
            for k, v in preds.items():
                ims.append(k.split("/")[-1])
                values.append(v)
                self.predicts[imPath] = v
            logging.warning(preds)
        
        mainClass = self.getWrongClasses()
        self.getEmptyPreds()
        
        return {"images":ims, self.out_dict[mainClass]:values}

    def get_preds(self, keywords,image ,path,  reader, platform : str, n_clusters : int = 10):
        predicts = {}
        #for path in paths:
        result = reader.readtext(image, min_size=4, low_text=0.1, link_threshold=0.1, add_margin=0.2)
        boxes = [np.array(item[0]) for item in result]
        texts = [item[1].lower() for item in result]

        clustered_boxes, clustered_texts = agglomerative_cluster_boxes(boxes, texts, n_clusters)
        if platform == 'vk':
            metrics = []
            for keyword in keywords[platform]:
                score, _, top_n_texts = get_top_n_boxes_texts(boxes, texts, keyword, min_score=70, n=4)
                metric = vk_metric(top_n_texts)
                metrics.append(metric)
            if metrics[0] > 20:
                predicts[path] = metrics[0]
            else:
                predicts[path] = max(metrics)
            
        if platform == 'tg':
            flag = True
            for keyword in keywords[platform]:
                score, _, top_n_texts = get_top_n_boxes_texts(boxes, texts, keyword, n=6)
                metric = tg_metric(top_n_texts)
                if metric != 0:
                    flag=False
                    predicts[path] = metric
                    break
            if flag:
                predicts[path] = 0
        if platform == 'yt':
            flag = True
            for keyword in keywords[platform]:
                score, _, top_n_texts = get_top_n_boxes_texts(boxes, texts, keyword, n=4)
                metric = yt1_metric(top_n_texts)
                if metric != 0:
                    flag=False                  
                    predicts[path] = metric
                    break
            if flag:
                predicts[path] = 0
        if platform == 'yt2':
            flag = True
            for keyword in keywords[platform]:
                score, _, top_n_texts = get_top_n_boxes_texts(boxes, texts, keyword, min_score=70, n=8)
                metric = yt2_metric(top_n_texts)
                if metric != 0:
                    flag=False
                    predicts[path] = metric
                    break
            if flag:
                predicts[path] = 0
        if platform == 'zn':
            flag = True
            for keyword in keywords[platform]:
                score, _, top_n_texts = get_top_n_boxes_texts(boxes, texts, keyword, n=5)
                metric = zn_metric(top_n_texts)
                if metric != 0:
                    flag=False
                    predicts[path] = metric
                    break
            if flag:
                predicts[path] = 0
        return predicts

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
        return mainClass
    def writeBroken(self, reason:str,images ):
        dfInvalidImages = {"Некорректные изображения":images,
                           "Причина": [reason]*len(images)}
        if os.path.isfile("temp/broken.csv"):
            df1 = pd.read_csv("temp/broken.csv")
            out = pd.concat([df1, pd.DataFrame(dfInvalidImages)])
            out.to_csv("temp/broken.csv", index=False)
        else:
            pd.DataFrame(dfInvalidImages).to_csv("temp/broken.csv", index=False)

    def drawBoxes(self, im,):

        result = self.Reader.readtext(im, min_size=4, low_text=0.1, link_threshold=0.1, add_margin=0.2)
        boxes = [np.array(item[0]) for item in result]
        texts = [item[1].lower() for item in result]
        n_clusters = 10
        clustered_boxes, clustered_texts = agglomerative_cluster_boxes(boxes, texts, n_clusters)

        visualize_clusters_on_image(im, clustered_boxes.values())
        """make plot with clusters and relations

        Args:
            img (np.ndarray):
            clustered_boxes (np.ndarray): list or np.ndarray in shape N, 4, 2 of N boxes

        Returns:
            np.ndarray
        """
        img = im.copy()
        for boxes in clustered_boxes:
            cluster_color = tuple(np.random.randint(0, 255, 3).tolist())

            central_box = boxes[0]
            central_centroid = compute_centroid(central_box)
            for box in boxes:
                centroid = compute_centroid(box)
                
                # Нанесение линии от центра кластера до каждого бокса
                cv2.line(img, tuple(map(int, centroid)), tuple(map(int, central_centroid)), cluster_color, 1, cv2.LINE_AA)

                # Нанесение бокса на изображение
                bbox = box_to_bbox(box)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), cluster_color, 1)
        return img

    def toTable(self, dataInferred):
        
        pass
    def get_classifier_prediction(self, image):
        im_ = cv2.resize(image, (512, 512), cv2.INTER_CUBIC)
        
        im_ = np.expand_dims(im_.T, 0).astype(np.float32)
        for i, channel in enumerate(im_):
            channel = (channel - channel.mean()) / (channel.std() + 1e-6)
            im_[i] = channel
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
        return "zn"
    return "error"



