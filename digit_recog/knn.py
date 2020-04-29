from collections import namedtuple
from typing import List

import numpy as np

train_data: List[np.array] = []
Neighbor = namedtuple('Neighbor', 'nclass distance')


def _euclidean_distance(instance1: np.array, instance2: np.array) -> float:
    if not len(instance1) == len(instance2):
        raise Exception('Exception euclid')
    return np.linalg.norm(instance1 - instance2)


def clear() -> None:
    global train_data
    train_data = []


def fit(*data: np.ndarray):
    global train_data
    train_data.extend(np.array(data))


def predict(instance: np.array, k: int = 5) -> int:
    global train_data
    neighbor_list = _get_neighbor_list(instance, train_data)
    sorted_distances = sorted(neighbor_list, key=lambda x: x.distance)
    class_dict = {}
    k_neighbors = sorted_distances[:k]
    for i in k_neighbors:
        n = class_dict.setdefault(i.nclass, 0)
        class_dict[i.nclass] = n + 1
    return max(class_dict, key=class_dict.get)


def _get_neighbor_list(instance: np.array, train_data: List[np.array]) -> List[Neighbor]:
    distances: List[Neighbor] = []
    for n_class in range(0, len(train_data)):
        for i in train_data[n_class]:
            neighbor = Neighbor(nclass=n_class, distance=_euclidean_distance(instance, i))
            distances.append(neighbor)
    return distances
