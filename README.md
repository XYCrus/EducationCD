# -- Statist Model
python leave_n_split.py ../data/wholedata.csv *n*

python statistic_training_complete.py ../model/training_dataset.csv ../result ../model

python statistic_prediction.py ../model/testing_dataset.csv ../result ../model

python statistic_performance_metrics.py ../result

# -- Deep learning
python train.py source.csv cpu epoch_number batch_number

*The column scorePercentage_pre of result/student_knowledge.csv is the prediction value*

python predict.py

# -- Leave One Test
python leave_n_test.py *d* *n*

* @*d*: int, specify dataset path, **"../data/wholedata.csv"** for whole dataset, **"../data/data2.csv"** for dataset of below average students

* @*n*: int, specify *n* unique dates to test on

# -- Pipeline Testing

* python statistic_training_pipeline.py ../data/wholedata.csv ../result ../model

* python statistic_training_pipeline.py ../data/wholedata.csv 