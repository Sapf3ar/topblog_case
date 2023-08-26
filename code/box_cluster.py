import numpy as np

"""
Сначала производится кластеризация текстбоксов, основанная на их взаимном расстоянии,
а пороговое значение зависит от высоты прямоугольника. Считается, что в рамках одной группы
текстбоксов они наклонены примерно одинаково.

Каждый кластер представлен в виде минимального прямоугольника, ограничивающего все
содержащиеся в нем текстбоксы.

После кластеризации в рамках каждого кластера производится сортировка текстбоксов
сверху-вниз слева-направо. После этого аналогичным методом сортируются и сами кластеры
(точнее ограничивающие их прямоугольники).

После всех сортировок возвращается правильная последовательность текстбоксов.
"""


def is_dot_left(a, b, c):
    # Лежит ли точка c слева от вектора ab
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]) < 0

def create_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

def eucl_dist(dot1, dot2):
    return np.linalg.norm(dot1 - dot2)

def is_perpendicular_in_segment(a, b, c):
    # Пересекает ли перпендикуляр из c к ab отрезок ab
    ab = b - a
    ac = c - a
    return 0 <= np.dot(ab, ac) <= np.dot(ab, ab)

def calc_perpendicular(a, b, c):
    # Расстояние от точки c к прямой ab
    return abs(np.cross(c - a, b - a)) / np.linalg.norm(b - a)


class BoxCluster:
    def __init__(self, box, tight_scale=3, dist_thresh=0.4):
        self.boxes = np.expand_dims(box, axis=0)
        self.tight_scale = tight_scale
        self.dist_threshold = np.linalg.norm(box[0] - box[-1]) * dist_thresh

        # Создается новая система координат
        # Точка отсчета - левая верхняя точка прямоугольника box[0]
        # Оси параллельны сторонам прямоугольника
        # Это нужно для ускорения подсчета расстояний между кластером
        # и прямоугольниками.
        self.bias = -box[0]
        top_side_vec = box[1] - box[0]
        top_side_vec_norm = top_side_vec / np.linalg.norm(top_side_vec)
        self.angle = np.arccos(np.dot([1, 0], top_side_vec_norm))
        if is_dot_left(box[0], box[0] + [1, 0], box[1]):
            self.angle *= -1

        self.rot_matrix = create_rotation_matrix(self.angle)
        self.rot_back_matrix = create_rotation_matrix(-self.angle)
        new_box = self.transform_box(box)
        self.bound_x = new_box[1][0]
        self.bound_y = new_box[3][1]


    def transform_dot(self, dot):
        return np.matmul(self.rot_matrix, dot + self.bias)

    def transform_box(self, box):
        return np.apply_along_axis(self.transform_dot, 1, box)

    def transform_dot_back(self, dot):
        return np.matmul(self.rot_back_matrix, dot) - self.bias

    def transform_box_back(self, box):
        return np.apply_along_axis(self.transform_dot_back, 1, box)

    def is_projections_intersect(self, x_projection, y_projection):
        if x_projection[0] > self.bound_x or y_projection[0] > self.bound_y or \
           x_projection[1] < 0 or y_projection[1] < 0:
            return False
        return True

    def calc_dist(self, box):
        cluster = self.get_cluster_boundaries()
        dist = np.Inf
        for dot1 in cluster:
            for dot2 in box:
                dist = min(dist, eucl_dist(dot1, dot2))

        for i in range(len(box)):
            dot1 = box[i]
            dot2 = box[(i + 1) % 4]
            for dot3 in cluster:
                if is_perpendicular_in_segment(dot1, dot2, dot3):
                    dist = min(dist, calc_perpendicular(dot1, dot2, dot3))

        for i in range(len(cluster)):
            dot1 = cluster[i]
            dot2 = cluster[(i + 1) % 4]
            for dot3 in box:
                if is_perpendicular_in_segment(dot1, dot2, dot3):
                    dist = min(dist, calc_perpendicular(dot1, dot2, dot3))
        return dist

    def is_box_near(self, box, x_projection, y_projection):
        if self.is_projections_intersect(x_projection, y_projection):
            return True
        return self.calc_dist(box) <= self.dist_threshold

    def get_box_projections(self, box):
        transformed_box = self.transform_box(box)
        x_projection = [np.min(transformed_box[:,0]), np.max(transformed_box[:,0])]
        y_projection = [np.min(transformed_box[:,1]), np.max(transformed_box[:,1])]
        return x_projection, y_projection

    def adjust_boundaries(self, x_projection, y_projection):
        bias_delta = np.array([0, 0])

        self.bound_x = max(self.bound_x, x_projection[1])
        if x_projection[0] < 0:
            bias_delta[0] = x_projection[0]
            self.bound_x -= x_projection[0]

        self.bound_y = max(self.bound_y, y_projection[1])
        if y_projection[0] < 0:
            bias_delta[1] = y_projection[0]
            self.bound_y -= y_projection[0]

        self.bias = -self.transform_dot_back(bias_delta)

    def add_if_near(self, box):
        x_projection, y_projection = self.get_box_projections(box)

        if self.is_box_near(box, x_projection, y_projection):
            self.adjust_boundaries(x_projection, y_projection)
            self.boxes = np.vstack((self.boxes, np.expand_dims(box, axis=0)))
            return True
        return False

    def get_cluster_boundaries(self):
        box = np.array([
            [0, self.bound_y],
            [self.bound_x, self.bound_y],
            [self.bound_x, 0],
            [0, 0]
        ])
        return self.transform_box_back(box)

    def merge_if_near(self, cluster):
        box = cluster.get_cluster_boundaries()
        x_projection, y_projection = self.get_box_projections(box)
        if self.is_box_near(box, x_projection, y_projection):
            self.adjust_boundaries(x_projection, y_projection)
            for box in cluster.boxes:
                self.boxes = np.vstack((self.boxes, np.expand_dims(box, axis=0)))
            return True
        return False

    def get_sorted_boxes(self):

        transform_with_source = lambda box: (self.transform_box(box), box)
        box_pairs = list(map(transform_with_source, self.boxes))
        boxes_sorted_by_y = sorted(box_pairs, key=lambda box_pair: box_pair[0][1][1])

        sorted_boxes = []
        start_box_pair = boxes_sorted_by_y[0]
        row = [start_box_pair[1]]
        y_min, y_max = start_box_pair[0][:,1].min(), start_box_pair[0][:,1].max()
        row_y_low_boundary = y_max - self.tight_scale * (y_max - y_min)

        for box_pair in boxes_sorted_by_y[1:]:
            if box_pair[0][:,1].min() > row_y_low_boundary:
                sorted_boxes += sorted(row, key=lambda box: box[0][0])
                row = [box_pair[1]]
                y_min, y_max = box_pair[0][:,1].min(), box_pair[0][:,1].max()
                row_y_low_boundary = y_max - self.tight_scale * (y_max - y_min)
            else:
                row.append(box_pair[1])

        sorted_boxes += sorted(row, key=lambda box: box[0][0])
        return sorted_boxes

def cluster_boxes(dt_boxes, tight_scale, dist_thresh):
    clusters = []
    for box in dt_boxes:
        is_added = False
        for cluster in clusters:
            if cluster.add_if_near(box):
                is_added = True
                break
        if not is_added:
            clusters.append(BoxCluster(box, tight_scale, dist_thresh))

    prev_size = -1

    while len(clusters) != prev_size:
        prev_size = len(clusters)
        for i in range(len(clusters)):
            merged_clusters_indexes = []
            for j in range(i + 1, len(clusters)):
                if clusters[i].merge_if_near(clusters[j]):
                    merged_clusters_indexes.append(j)

            if merged_clusters_indexes:
                for del_index in reversed(merged_clusters_indexes):
                    del clusters[del_index]
                break

    return sorted(clusters, key=lambda cluster: -cluster.bias[1])



def box2key(box):
    return ",".join(np.array(box).astype(int).flatten().astype(str))

def get_clusters(boxes, texts, tight_scale=0.3, dist_thresh=1):
    clusters = []
    boxes_map = dict()
    for box, text in zip(boxes, texts):
        boxes_map[box2key(box)] = text

    
    box_clusters = cluster_boxes(boxes, tight_scale=tight_scale, dist_thresh=dist_thresh)
    
    for box_cluster in box_clusters:
        box_text_cluster = []
        used = set()
        for box in box_cluster.get_sorted_boxes():
            key = box2key(box)
            if key in used:
                continue
            used.add(key)
            text = boxes_map[key]
            box_text_cluster.append([box, text])
        clusters.append(box_text_cluster)

    return clusters
