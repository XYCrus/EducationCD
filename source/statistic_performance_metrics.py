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
    accuracy = round(accu_purified.mean(), 5)

    #mse
    diff = np.array(df["differences"])
    diff_purified = np.array([i for i in diff if not np.isnan(i)])
    sum_square = sum(diff_purified ** 2)
    mse = round(sum_square / len(df), 5)

    # all type accuracy
    all_type_sum = df["accuracyFlag"].groupby(df["questionType"]).sum()
    all_type_count = df["accuracyFlag"].groupby(df["questionType"]).count()
    all_type_sum = dict(all_type_sum)
    all_type_count = dict(all_type_count)
    all_type_accuracy = {}
    for key1, value1 in all_type_sum.items():
        for key2, value2 in all_type_count.items():
            if key1 == key2:
                all_type_accuracy[key1] = round(value1 / value2, 5)

    # all type mse
    df["squared_differences"] = df["differences"] ** 2
    all_type_squared_diff = df["squared_differences"].groupby(df["questionType"]).sum()
    all_type_squared_diff = dict(all_type_squared_diff)
    all_type_mse = {}
    for key1, value1 in all_type_squared_diff.items():
        for key2, value2 in all_type_count.items():
            if key1 == key2:
                all_type_mse[key1] = round(value1 / value2, 5)

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

    with open('../result/statistic_performance_metrics.csv', 'w', newline = '') as csvfile:
        my_writer = csv.writer(csvfile)
        my_writer.writerow(["startDatetime", "Total_acc", "Total_mse", "FILLBLANK_acc", "FILLBLANK_mse", "MCHOICE_acc", "MCHOICE_mse", "SCHOICE_acc", "SCHOICE_mse", "SHORTANSWER_acc", "SHORTANSWER_mse"])

        for i in date_list:
            list_metrics = list(stat_metrics(pred_df_copy[pred_df_copy['startDatetime'] == i]))
            keys = list(list_metrics[2].keys())
            result = [i, list_metrics[0], list_metrics[1]]
            if 'FILLBLANK' in keys:
                result.append(list_metrics[2]['FILLBLANK'])
                result.append(list_metrics[3]['FILLBLANK'])
            else:
                result.append(None)
                result.append(None)
            if 'MCHOICE' in keys:
                result.append(list_metrics[2]['MCHOICE'])
                result.append(list_metrics[3]['MCHOICE'])
            else:
                result.append(None)
                result.append(None)
            if 'SCHOICE' in keys:
                result.append(list_metrics[2]['SCHOICE'])
                result.append(list_metrics[3]['SCHOICE'])
            else:
                result.append(None)
                result.append(None)
            if 'SHORTANSWER' in keys:
                result.append(list_metrics[2]['SHORTANSWER'])
                result.append(list_metrics[3]['SHORTANSWER'])
            else:
                result.append(None)
                result.append(None)
            my_writer.writerows([result])
        