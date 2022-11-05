import sys
import os.path
import pandas as pd
import datetime
from statistic_training import create_statistic_model

def retrain(newdatapath, result_folder = '../result', oldfolder = '../model', newfolder = '../new_model'):
    olddatapath = oldfolder + '/knowledge_dataset.csv'
    df = pd.read_csv(olddatapath)
    new = pd.read_csv(newdatapath)

    # mark the new data as flag == 1
    new['flag'] = 1

    # reformat the new df in accordance with the old one
    new['startDatetime'] = new['startDatetime'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%y %H:%M'))

    # concat the new data to old knowledge dataset
    df = pd.concat([df,new], ignore_index=True)

    # sort the new knowledge dataset by stu id
    df = df.sort_values(by = 'stuUserId', ascending=True)

    # write new knowledge dataset to new_model folder
    if not os.path.exists(newfolder):
        os.mkdir(newfolder)
    df.to_csv(newfolder+"/knowledge_dataset.csv", index=False)

    # rebuild statistic model to new folder
    create_statistic_model(newfolder + "/knowledge_dataset.csv", newfolder, result_folder)

    
if __name__ == '__main__':

    newdatapath = sys.argv[1]
    result_folder = sys.argv[2]
    old_folder = sys.argv[3]
    new_folder = sys.argv[4]
    if not newdatapath.endswith('.csv'):
        print('wrong file type')
        exit(1)

    retrain(newdatapath, result_folder, old_folder, new_folder)
    