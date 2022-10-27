import sys
import pandas as pd
import datetime
import json


def collect_pairs(datapath):
    df = pd.read_csv(datapath)
    pair_set = set([])
    for i in range(df.shape[0]):
        stu = df['stuUserId'].iloc[i]
        knowledges = json.loads(df['knowledgeTagIds'].iloc[i])
        for klg in knowledges:
            pair_set.add((stu,klg))
    return pair_set


def build_dataset(datapath, wholedata_path, n=3):
    # n: number of records to find for each missing pair

    wholedata = pd.read_csv(wholedata_path)
    data = pd.read_csv(datapath)
    
    # flag the current data as 1
    data['flag'] = 1

    # find the missed pairs
    pairs = collect_pairs(wholedata_path) - collect_pairs(datapath)

    # sort the wholedataset by exam time
    wholedata['startDatetime'] = wholedata['startDatetime'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%y %H:%M'))
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
                    if len(single[pair]) == n: 
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
        while len(record) < n:
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

    # write into file latest_234+.csv
    data.to_csv("../data/latest_234+.csv")



if __name__ == '__main__':
    # sys.argv catches the command parameters typed (sep with ' ')
    # read in and record parameters

    # the first component of the string typed
    data_file = sys.argv[1]
    wholedata_file = sys.argv[2]
    if not (data_file.endswith('.csv') and wholedata_file.endswith('.csv')):
        print('wrong file type')
        exit(1)
    
    if len(sys.argv) == 4:
        n = sys.argv[3]

    build_dataset(data_file, wholedata_file, n=3)
