# -- Deep learning
python train.py source.csv cpu epoch_number batch_number

*The column scorePercentage_pre of result/student_knowledge.csv is the prediction value*

python predict.py 

# -- Statist Model
python pipeline_activation.py *n* *d*

* @*n*: int, specify *n* unique dates to test on

* @*d*: int, specify dataset path, **"../data/wholedata.csv"** for whole dataset, **"../data/data2.csv"** for dataset of below average students