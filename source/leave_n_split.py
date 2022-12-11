#%% packages
import sys
import pandas as pd
import numpy as np
import datetime
import json
import os

#%% functions
def leave_n_pipeline(wholedata_path, leave_n = 1):

    wholedata = pd.read_csv(wholedata_path)

    datapath = "../model/training_dataset.csv" 
    testpath = "../model/testing_dataset.csv"

    exam_dates = np.sort(wholedata['startDatetime'].unique())
    test_dates = exam_dates[-int(leave_n):]

    trainset = wholedata[~wholedata['startDatetime'].isin(test_dates)]
    testset = wholedata[wholedata['startDatetime'].isin(test_dates)]

    trainset.to_csv(datapath, index = False)
    testset.to_csv(testpath, index = False)

def check_folder():
    if not os.path.exists('../model'):
        os.mkdir('../model')

#%% main
if __name__ == '__main__':

    original_dataset = sys.argv[1]
    leave_n = 1 
    
    if not (original_dataset.endswith('.csv')):
        print('wrong file type')
        exit(1)
    
    if len(sys.argv) == 3:
        leave_n = sys.argv[2]

    check_folder()
    leave_n_pipeline(original_dataset, leave_n)
