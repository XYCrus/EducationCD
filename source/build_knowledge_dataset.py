import sys
import pandas as pd
import numpy as np
import datetime
import json
import os


def collect_pairs(datapath):
    df = pd.read_csv(datapath)
    pair_set = set([])
    for i in range(df.shape[0]):
        stu = df['stuUserId'].iloc[i]
        knowledges = json.loads(df['knowledgeTagIds'].iloc[i])
        for klg in knowledges:
            pair_set.add((stu,klg))
    return pair_set


def build_latest(wholedata_path, n=3):
    # n: number of latest exams to select

    wholedata = pd.read_csv(wholedata_path)

    # check the number of exams
    exam_count = wholedata['startDatetime'].unique()
    if exam_count <= n:
        wholedata.to_csv("../model/latest_dataset.csv")
    else: 
        # pick the dates of last n exams
        wholedata['startDatetime'] = wholedata['startDatetime'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%y %H:%M'))
        exam_dates = np.sort(wholedata['startDatetime'].unique())
        latest_dates = exam_dates[-n:]
        # pick the data where dates in the last dates
        latest = wholedata[wholedata['startDatetime'].isin(latest_dates)]
        # write into file
        latest.to_csv("../model/latest_dataset.csv")


def build_dataset(wholedata_path, n_latest = 3, n_fill = 3):
    # n_latest: number of latest exams to select
    # n_fill: number of records to find for each missing pair

    # build latest dataset

    wholedata = pd.read_csv(wholedata_path)
    datapath = "../model/latest_dataset.csv" #path for latest dataset

    ## check the number of exams
    ### if < n_latest, only create latest dataset
    exam_count = wholedata['startDatetime'].unique().size
    if exam_count <= n_latest:
        wholedata.to_csv("../model/latest_dataset.csv")
        return None

    ### elif > n_latest, create both latest dataset and knowledge dataset 
    ## pick the dates of last n exams
    wholedata['startDatetime'] = wholedata['startDatetime'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%y %H:%M'))
    exam_dates = np.sort(wholedata['startDatetime'].unique())
    latest_dates = exam_dates[-n_latest:]
    # pick the data where dates in the last dates
    latest = wholedata[wholedata['startDatetime'].isin(latest_dates)]
    # write into file
    latest.to_csv(datapath)

    # build knowledge dataset
    data = latest
    
    # flag the current data as 1
    data['flag'] = 1

    # find the missed pairs
    pairs = collect_pairs(wholedata_path) - collect_pairs(datapath)

    # sort the wholedataset by exam time
    wholedata = wholedata.sort_values(by = 'startDatetime', ascending=False)

    # iterate the wholedataset, recording indices of single and complex appearances of each pair in pairs
    single = {pair:[] for pair in pairs}
    complex = {pair:[] for pair in pairs}

    N = wholedata.shape[0]
    for i in range(N):
        stu = wholedata['stuUserId'].iloc[i]
        klgs = json.loads(wholedata['knowledgeTagIds'].iloc[i])
        for klg in klgs:
            pair = (stu,klg)
            if pair in pairs:
                if len(klgs) == 1:
                    single[pair].append(wholedata.index[i])
                    ## the pair is no longer interested if n single appearance records have been collected
                    if len(single[pair]) == n_fill: 
                        pairs.discard(pair)
                else:
                    complex[pair].append(wholedata.index[i])
        if len(pairs) == 0: 
            break
    
    # integrate single and complex records
    ## if there is already n records in single, discord complex records
    ## else, use first m complex records to fill
    records = {}
    pairs = single.keys()
    for pair in pairs:
        record = single[pair]
        while len(record) < n_fill:
            if complex[pair]:
                record.append(complex[pair].pop(0))
            else:
                break
        records[pair] = record
    
    # extract all the indices to add
    indices = set([])
    for record in records.values():
        for index in record:
            indices.add(index)
    indices = list(indices)
    
    # append to data the rows of missing pairs corresponding to indices, with flag = 0
    newdata = wholedata.loc[indices]
    newdata['flag'] = 0
    data = data.append(newdata)

    # write into csv file
    data.to_csv("../model/knowledge_dataset.csv")


def check_folder():
    if not os.path.exists('../model'):
        os.mkdir('../model')


if __name__ == '__main__':
    # sys.argv catches the command parameters typed (sep with ' ')
    # read in and record parameters

    # the first component of the string typed
    wholedata_file = sys.argv[1]
    if not (wholedata_file.endswith('.csv')):
        print('wrong file type')
        exit(1)
    
    if len(sys.argv) >= 3:
        n_latest = sys.argv[2]
    
    if len(sys.argv) >= 4:
        n_fill = sys.argv[3]

    check_folder()
    build_dataset(wholedata_file, n_latest=3, n_fill=3)
