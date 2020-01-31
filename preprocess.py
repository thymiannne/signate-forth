# !usr//bin/env python
# -*- coding:utf-8 -*-

import datetime
import csv
import numpy as np
import pandas as pd
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

START_DATE = datetime.datetime(2017, 4, 1)
TRACK_INDEXES = ["date", "キロ程", "高低左", "速度"]
EQUIPMENT_INDEXES = ["キロ程", "バラスト", "ロングレール", "マクラギ種別",
                     "橋りょう", "踏切", "通トン", "曲線半径", "フラグ"]
DATA_INDEXES = ["date", "キロ程", "速度", "バラスト", "ロングレール",
                "マクラギ種別", "橋りょう", "踏切", "通トン", "曲線半径"]  # 列数が17になっているべき

SPEED_MEANS = {
    'A': 75.01464201541873,
    'B': 67.15571510061953,
    'C': 89.372625487744,
    'D': 74.0672439862501,
}

ROUTE = ['A', 'B', 'C', 'D']
DATE_RANGE = range(365, 456)
KILO_RANGES = {
    'A': range(1, 37906),
    'B': range(1, 31531),
    'C': range(1, 65682),
    'D': range(1, 25691),
}


class Track(Enum):
    DATE = 0
    KILO = 1
    LEFT = 2
    SPEED = 8


class Equipment(Enum):
    KILO = 0
    BALLAST = 1
    LONG_RAIL = 2
    SLEEPER = 3
    BRIDGE = 4
    CROSSING = 5
    PASSAGE = 6
    CURVATURE = 7
    FLAG = 8


def cast_safely(val, cast_type=float):
    try:
        return cast_type(val)
    except:
        return cast_type()


class PreProcessor:
    @classmethod
    def get_train_data(cls, route):
        # tracks = pd.read_csv(f"../dataset/track_{route}.csv", parse_dates=["date"])
        # equipments = pd.read_csv(f"../dataset/equipment_{route}.csv")
        X, Y = [], []
        with open(f"../dataset/track_{route}.csv", encoding="utf_8") as track_file:
            csv_dict_reader = csv.DictReader(track_file)
            equipment = pd.read_csv(f"../dataset/equipment_{route}.csv")
            for i, row in enumerate(csv_dict_reader):
                # if i > limit: break
                x = []
                kilotei = int(row["キロ程"])
                index = kilotei - 10000
                if equipment.iat[index, 8]:  # フラグが立っていれば
                    continue  # 学習データには加えない
                try:
                    Y.append(float(row["高低左"]))
                except:
                    continue
                date = datetime.datetime.strptime(row['date'], '%Y-%m-%d')
                day = (date - START_DATE).days
                x.append(day)
                x.append(kilotei)
                """try:
                    x.append(float(row['速度']))
                except:
                    x.append(SPEED_MEANS[route])"""
                for j, equip in enumerate(equipment.iloc[index, 1:8]):
                    if j == 2:
                        x.extend(vectorize_sleepers(int(equip)))
                    else:
                        x.append(equip)
                X.append(x)

        X = np.array(X)
        Y = np.array(Y)
        # print(f'X shape: {X.shape}, Y shape: {Y.shape}')
        # X_train, X_test, y_train, y_test = train_test_split(X, Y)
        # print(f'train size: {y_train.size}, test size: {y_test.size}')
        # return X_train, X_test, y_train, y_test
        return X, Y

    @classmethod
    def get_test_data(cls, route):
        equipment = pd.read_csv(f"../dataset/equipment_{route}.csv")
        X = []
        # with open('index_master.csv', encoding='utf-8') as master:
        # reader = csv.DictReader(master)
        for date in DATE_RANGE:
            for kilo in KILO_RANGES[route]:
                x = [date, kilo]
                index = kilo - 10000
                series = equipment.iloc[index, 1:8]
                for j, equip in enumerate(series):
                    if j == 2:
                        x.extend(vectorize_sleepers(int(equip)))
                    else:
                        x.append(equip)
                X.append(x)
        return np.array(X)

    @classmethod
    def get_test_data_for_few(cls, kilo, series):
        X = []
        for date in DATE_RANGE:
            x = [date, kilo]
            for j, equip in enumerate(series):
                if j == 2:
                    x.extend(vectorize_sleepers(int(equip)))
                else:
                    x.append(equip)
            X.append(x)
        return X

    @classmethod
    def get_perkilo_train_data(cls, perkilo_track):
        data = np.array(perkilo_track.loc[:, ['date', '高低左']])
        delete_rows = []
        for i, x in enumerate(data):
            if np.isnan(x[1]):
                delete_rows.append(i)
                continue
            x[0] = (x[0] - pd.Timestamp('2017-04-1 00:00:00')).days
        data = np.delete(data, delete_rows, 0)
        X, Y = data[:, 0].reshape(-1, 1), data[:, 1]
        return X, Y

    @classmethod
    def get_perkilo_test_data(cls):
        return np.array(DATE_RANGE).reshape(-1, 1)


def vectorize_sleepers(num):
    vec = [0 for i in range(8)]
    print()
    if num is not None:
        vec[num - 1] = 1
    return vec


if __name__ == '__main__':
    X_train, y_train = PreProcessor.get_train_data('A')
    # print(np.array(X_train))
    # print(np.array(y_train))
