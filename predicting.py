#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv
import pickle

from scripts.preprocess import PreProcessor
from scripts.train import Model

ROUTE = ['A', 'B', 'C', 'D']
DATE_RANGE = range(365, 456)
KILO_RANGES = {
    'A': range(10000, 37906),
    'B': range(10000, 31531),
    'C': range(10000, 65684),
    'D': range(10000, 25691),
}

THRESHOLDS = {
    'A': 250,
    'B': 150,
    'C': 200,
    'D': 100,
}

OFFSETS = {
    'A': 0,
    'B': 2539446,
    'C': 4498767,
    'D': 9566011,
}


def unzip(li):
    return map(iter, zip(*li))


def predict_main(filename):
    with open(filename, mode='w', encoding='utf-8', newline='') as result:
        writer = csv.writer(result)
        row = 0
        for route in ROUTE:
            print(f'now writing for {route}...')
            X_train, y_train = PreProcessor.get_train_data(route)
            # clf = pickle.load(open(f'xgbregressor_{route}', 'rb'))
            clf = Model.naive_train_with_xgboost(X_train, y_train)
            # Model.evaluate(clf, X_test, y_test)
            # pickle.dump(clf, open(f'xgbregressor_{route}', 'wb'))
            X = PreProcessor.get_test_data(route)
            Y = clf.predict(X)  # clfはroute毎に分ける
            # Y.flatten()
            for value in Y:
                writer.writerow([str(row), str(value)])
                row += 1


def predict_main_(filename):
    with open(filename, mode='w', encoding='utf-8', newline='') as result:
        writer = csv.writer(result)
        row = 0
        for route in ROUTE:
            print(f'now writing for {route}...')
            track = pd.read_csv(f"../dataset/track_{route}.csv", parse_dates=["date"])
            equipment = pd.read_csv(f"../dataset/equipment_{route}.csv")
            X_whole_train, y_whole_train = PreProcessor.get_train_data(route)
            whole_clf = Model.naive_train_with_xgboost(X_whole_train, y_whole_train)
            print('Whole training finished!!!')
            pred_values = []
            for kilo in KILO_RANGES[route]:
                perkilo_track = track[track["キロ程"] == kilo]
                X_train, Y_train = PreProcessor.get_perkilo_train_data(perkilo_track)
                if X_train.shape[0] >= THRESHOLDS[route]:
                    clf = Model.naive_train_with_xgboost(X_train, Y_train)
                    # clf = gridsearch_train_with_xgboost(X_train, y_train)
                    # Model.evaluate(clf, X_test, y_test)
                    X_test = PreProcessor.get_perkilo_test_data()
                    Y_pred = clf.predict(X_test)  # clfはroute毎に分ける
                else:
                    index = kilo - 10000
                    series = equipment.iloc[index, 1:8]
                    X_test = PreProcessor.get_test_data_for_few(kilo, series)
                    Y_pred = whole_clf.predict(X_test)
                pred_values.append(Y_pred)

            # row = OFFSETS[route]
            for values in unzip(pred_values):
                for value in values:
                    writer.writerow([str(row), str(value)])
                    row += 1


def check_master():
    with open('../dataset/index_master.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for route in ROUTE:
            while True:
                row = next(reader)
                if row['路線'] != route:
                    print(f'Route {route} finished.')
                    print(dict(row))
                    break
        try:
            rest = next(reader)
            print(rest)
        except StopIteration:
            print('Its OK.')


def check_master_(filename):
    with open(filename, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for route, ids in zip(ROUTE, [2539445, 4498766, 9566010, 10000000]):
            while True:
                row = next(reader)
                # if row['路線'] != route:
                if row['id'] == str(ids):
                    print(f'Route {route} finished.')
                    print(dict(row))
                    break
        try:
            rest = next(reader)
            print(rest)
        except StopIteration:
            print('Its OK.')


if __name__ == '__main__':
    predict_main('result_submit_whole.csv')
    # predict_main_('result_submit_x_x.csv')
    # check_master_('../dataset/index_master.csv')
    # check_master_('../dataset/index_master.csv')
