#%%
import sys
from datetime import datetime
from build_knowledge_dataset import build_dataset
from statistic_training import create_statistic_model, check_folder
from continuous_training import retrain

#%%
if __name__ == '__main__':
    begin = datetime.now()

    if len(sys.argv) <= 4:  # no retrain
        wholedata_file = sys.argv[1]
        if not (wholedata_file.endswith('.csv')):
            print('wrong file type')
            exit(1)

        result_folder = sys.argv[2]
        model_folder = sys.argv[3]

        check_folder(result_folder=result_folder, model_folder=model_folder)
        build_dataset(wholedata_file, model_folder)
        data_file = model_folder + "/knowledge_dataset.csv"
        create_statistic_model(data_file, model_folder=model_folder, result_folder=result_folder)


    elif len(sys.argv) >= 5:  # retrain
        new_file = sys.argv[1]
        result_folder = sys.argv[2]
        old_folder = sys.argv[3]
        new_folder = sys.argv[4]
        if not new_file.endswith('.csv'):
            print('wrong file type')
            exit(1)
        check_folder(model_folder=new_folder, result_folder=result_folder)
        retrain(new_file, result_folder, old_folder, new_folder)

    end = datetime.now()
    print("time: ", end - begin)
