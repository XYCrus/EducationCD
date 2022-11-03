import sys
import os.path
import pandas as pd
from statistic_training import create_statistic_model

def retrain(newdatapath, oldfolder = '../model', newfolder = '../new_model'):
    olddatapath = oldfolder + '/knowledge_datast.csv'
    df = pd.read_csv(olddatapath)
    new = pd.read_csv(newdatapath)

    # mark the new data as flag == 1
    new['flag'] = 1

    # concat the new data to old knowledge dataset
    df = pd.concat([df,new], ignore_index=True)

    # sort the new knowledge dataset by stu id
    df = df.sort_values(by = 'stu_user_id', ascending=True)

    # write new knowledge dataset to new_model folder
    if not os.path.exists(newfolder):
        os.mkdir(newfolder)
    df.to_csv(newfolder+"/knowledge_dataset.csv")

    # rebuild statistic model to new folder
    create_statistic_model(newfolder + "/knowledge_dataset.csv", newfolder)

    
if __name__ == '__main__':

    newdatapath = sys.argv[1]
    oldfolder = sys.argv[2]
    newfolder = sys.argv[3]
    if not newdatapath.endswith('.csv'):
        print('wrong file type')
        exit(1)

    retrain(newdatapath, oldfolder, newfolder)
    