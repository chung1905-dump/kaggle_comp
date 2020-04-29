import csv
from timeit import default_timer as timer
from typing import List

import knn
import pandas as pd
import numpy as np


def read_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


def get_train_data() -> List[pd.DataFrame]:
    ret = [None] * 10
    data = read_data('./data/train.csv')
    labels = data['label'].unique().tolist()

    for label in labels:
        # ret[label] = data.query(expr='label == ' + str(label))
        ret[label] = data.query(expr='label == ' + str(label)).drop(columns='label')

    return ret


def write_to_csv(data: List[list], name: str = 'result.csv', header: List[str] = ['ImageId', 'Label']) -> None:
    opener = open(name, 'w')
    writer = csv.writer(opener)
    writer.writerow(header)
    for datum in data:
        print(datum)
        writer.writerow(datum)
    opener.close()


def get_test_data() -> pd.DataFrame:
    return read_data('./data/test.csv')


def run() -> None:
    knn.clear()
    key = 0

    for datum in get_train_data():
        print('Fit class ' + str(key) + '...')
        knn.fit(datum.values)
        key += 1

    test_df = get_test_data()
    results = []
    img_id: int = 0

    for test_list in test_df.values:
        start = timer()
        img_id += 1
        results.append([img_id, knn.predict(np.array(test_list))])
        end = timer()
        print('Done ' + str(img_id) + ' in ' + str(end - start) + 's')

    write_to_csv(results)


run()
