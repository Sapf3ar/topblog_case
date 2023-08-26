import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans


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
