-- Deep learnring
python train.py source.csv cpu 1
python predict.py 

--statist model
python statistic_training.py ../data/latest_234.csv
python statistic_prediction.py ../model/model.csv ../data/prediction_set.csv


