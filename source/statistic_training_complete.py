import sys
import os
import string
import pandas as pd
import numpy as np
import datetime
import json
from build_klg_dataset import build_dataset
from statistic_training import create_statistic_model, check_folder

if __name__ == '__main__':
    
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
    data_file = "../model/knowledge_dataset.csv"
    create_statistic_model(data_file)
