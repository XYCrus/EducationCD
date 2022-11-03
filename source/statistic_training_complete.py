import sys
from build_knowledge_dataset import build_dataset
from statistic_training import create_statistic_model, check_folder
from continuous_training import retrain


if __name__ == '__main__':

    if len(sys.argv) <= 3: # no retrain
        wholedata_file = sys.argv[1]
        if not (wholedata_file.endswith('.csv')):
            print('wrong file type')
            exit(1)
        if len(sys.argv) == 2:
            check_folder()
            build_dataset(wholedata_file, n_latest=3, n_fill=3)
            data_file = "../model/knowledge_dataset.csv"
            create_statistic_model(data_file)

        if len(sys.argv) == 3:
            folder = sys.argv[2]
            check_folder(folder)
            build_dataset(wholedata_file, folder)
            data_file = folder + "/knowledge_dataset.csv"
            create_statistic_model(data_file, folder)

    
    elif len(sys.argv) >= 4: # retrain
        new_file = sys.argv[1]
        old_folder = sys.argv[2]
        new_folder = sys.argv[3]
        if not new_file.endswith('.csv'):
            print('wrong file type')
            exit(1)
        
        retrain(new_file, old_folder, new_folder)
        

    
