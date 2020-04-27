import math
from collections import namedtuple
from typing import List

train_data: list = []
Neighbor = namedtuple('Neighbor', 'nclass distance')


def _euclidean_distance(instance1: list, instance2: list) -> float:
    if not len(instance1) == len(instance2):
        raise Exception('Exception euclid')
    distance: float = 0
    for x in range(0, len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def clear() -> None:
    global train_data
    train_data = []


def fit(*data: List[list]):
    global train_data
    train_data.extend(data)


def predict(instance: list, k: int = 1) -> int:
    global train_data
    neighbor_list = _get_neighbor_list(instance, train_data)
    sorted_distances = sorted(neighbor_list, key=lambda x: x.distance)
    class_dict = {}
    k_neighbors = sorted_distances[:k]
    for i in k_neighbors:
        n = class_dict.setdefault(i.nclass, 0)
        class_dict[i.nclass] = n + 1
        # print('Distance: %f - Class: %i' % (i.distance, i.nclass))
    return max(class_dict, key=class_dict.get)


def _get_neighbor_list(instance, train_data) -> list:
    distances: list = []
    for n_class in range(0, len(train_data)):
        for i in train_data[n_class]:
            neighbor = Neighbor(nclass=n_class, distance=_euclidean_distance(instance, i))
            distances.append(neighbor)
    return distances
