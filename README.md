# -- Deep learning
python train.py source.csv cpu epoch_number batch_number

*The column scorePercentage_pre of result/student_knowledge.csv is the prediction value*

python predict.py 

# -- Statist Model
python leave_n_test.py *n* *d*

* @*n*: int, specify *n* unique dates to test on

* @*d*: int, specify dataset path, **"../data/wholedata.csv"** for whole dataset, **"../data/data2.csv"** for dataset of below average students

# -- Statist Model Manually
python leave_n_split.py ../data/wholedata.csv *n*

python statistic_training_complete.py ../model/training_dataset.csv ../result ../model

python statistic_prediction.py ../model/testing_dataset.csv ../result ../model

python statistic_performance_metrics.py ../result *n*