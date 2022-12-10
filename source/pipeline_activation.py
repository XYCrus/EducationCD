import os
import sys
import numpy as np
import csv

if __name__ == '__main__':
    iteration = sys.argv[1]
    dataset_option = sys.argv[2]
    
    for walk in np.arange(1, int(iteration) + 1):
        os.system("python leave_n_split.py {} {}".format(dataset_option, walk))
        os.system("python statistic_training_complete.py ../model/training_dataset.csv ../result ../model")
        os.system("python statistic_prediction.py ../model/testing_dataset.csv ../result ../model")
        os.system("python statistic_performance_metrics.py ../result {}".format(walk))
        





