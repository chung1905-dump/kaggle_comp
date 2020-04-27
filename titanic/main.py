import csv
from typing import List

import pandas as pd
from pandas._typing import FrameOrSeries

import knn


def read_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


def process_data(data: pd.DataFrame, is_test_data: bool = False) -> pd.DataFrame:
    columns = ['id', 'survived', 'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin',
               'embarked']
    if is_test_data:
        columns.remove('survived')
    data.columns = columns

    # New column
    data['related_people'] = data['sibsp'] + data['parch']

    # Transform column
    data['sex'] = data['sex'].apply(lambda x: 0 if x == 'female' else 1)
    data['embarked'] = data['embarked'].apply(lambda x: 0 if x == 'C' else (1 if x == 'S' else 2))

    # Fill NaN
    data = data.fillna(value=0)

    # Remove unused columns
    labels = ['name', 'sibsp', 'parch', 'cabin', 'ticket', 'id']
    if is_test_data:
        labels.remove('id')
    data = data.drop(axis=1, labels=labels)

    return data


def classify_train_data(all_data: pd.DataFrame) -> List[pd.DataFrame]:
    ret_arr = [
        all_data.query(expr='survived == 0').drop(columns='survived'),
        all_data.query(expr='survived == 1').drop(columns='survived'),
    ]
    return ret_arr


def get_train_data() -> List[pd.DataFrame]:
    data = read_data('train.csv')
    data = process_data(data)
    data = classify_train_data(data)
    return data


def get_test_data() -> pd.DataFrame:
    data = read_data('test.csv')
    return process_data(data, is_test_data=True)


def run() -> None:
    test_data = get_test_data()
    data = get_train_data()
    knn.clear()
    knn.fit(data[0].values.tolist(), data[1].values.tolist())
    test_list: list = test_data.values.tolist()
    results = []
    for i in test_list:  # type: list
        id = i[:1][0]
        del i[0]
        results.append([int(id), knn.predict(i)])
    write_to_csv(results)


def write_to_csv(data: List[list], name: str = 'result.csv', header: List[str] = ['PassengerId','Survived']) -> None:
    opener = open(name, 'w')
    writer = csv.writer(opener)
    writer.writerow(header)
    for datum in data:
        print(datum)
        writer.writerow(datum)
    opener.close()


def get_sample_data(data: pd.DataFrame, n: int = 1) -> tuple:
    sample = data.sample(n)
    return data.drop(sample.index), sample


def run_with_sample() -> None:
    total_point = 0.0
    number_test = 100
    for i in range(0, number_test):
        data = get_train_data()
        samples: List[FrameOrSeries] = []
        data[0], sample = get_sample_data(data[0], 25)  # type: FrameOrSeries
        samples.append(sample)
        data[1], sample = get_sample_data(data[1], 25)  # type: FrameOrSeries
        samples.append(sample)

        knn.clear()
        knn.fit(data[0].values.tolist(), data[1].values.tolist())

        point = 0
        total = 0
        for k in range(0, len(samples)):
            for s in samples[k].values:
                if knn.predict(s.tolist(), k=3) == k:
                    point += 1
                total += 1

        print('point: ' + str(point))
        print('total: ' + str(total))
        total_point += float(point / total * 100)
        print(point / total * 100)

    print('total_point: ' + str(total_point / number_test))


run()
