#%% packages
import pandas as pd
import numpy as np
from collections import Counter
import sys
import os
import csv

#%% file path
result_name = '/result.csv'

#%% mse function
def stat_metrics(df):
    #accuracy
    accu = np.array(df["accuracyFlag"])
    accu_purified = np.array([i for i in accu if i is not None])
    accuracy = accu_purified.mean()

    #mse
    diff = np.array(df["differences"])
    diff_purified = np.array([i for i in diff if not np.isnan(i)])
    sum_square = sum(diff_purified ** 2)
    mse = sum_square / len(df)

    # all type accuracy
    all_type_sum = df["accuracyFlag"].groupby(df["questionType"]).sum()
    all_type_count = df["accuracyFlag"].groupby(df["questionType"]).count()
    all_type_sum = dict(all_type_sum)
    all_type_count = dict(all_type_count)
    all_type_accuracy = {}
    for key1, value1 in all_type_sum.items():
        for key2, value2 in all_type_count.items():
            if key1 == key2:
                all_type_accuracy[key1] = value1 / value2

    # all type mse
    df["squared_differences"] = df["differences"] ** 2
    all_type_squared_diff = df["squared_differences"].groupby(df["questionType"]).sum()
    all_type_squared_diff = dict(all_type_squared_diff)
    all_type_mse = {}
    for key1, value1 in all_type_squared_diff.items():
        for key2, value2 in all_type_count.items():
            if key1 == key2:
                all_type_mse[key1] = value1 / value2

    return accuracy, mse, all_type_accuracy, all_type_mse

#%% main
if __name__ == '__main__':
    result_path = sys.argv[1]
    walk = sys.argv[2]

    df = pd.read_csv(result_path + result_name)
    pred_df_copy = df.copy()

    date_list = pred_df_copy['startDatetime'].unique()

    if not os.path.exists('../result'):
        os.mkdir('../result')

    with open('../result/statistic_performance_metrics.csv', 'a', newline = '') as csvfile:
        my_writer = csv.writer(csvfile)
        my_writer.writerow(["--------------ITERATION {}--------------".format(walk)])

        for i in date_list:
            list_metrics = list(stat_metrics(pred_df_copy[pred_df_copy['startDatetime'] == i]))
            list_metrics.insert(0, i)
            my_writer.writerows(map(lambda x: [x], list_metrics))
        