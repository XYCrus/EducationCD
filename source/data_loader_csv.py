import json
import torch
import csv
import numpy as np
import os


def qt_transformer(qt):
    if qt == 'FILLBLANK':
        return 1
    elif qt == 'MCHOICE':
        return 2
    elif qt == 'SCHOICE':
        return 3
    elif qt == 'SHORTANSWER':
        return 4


class TrainDataLoaderCSV(object):
    """
    data loader for training
    """

    def __init__(self, file, total_epoch, batch_s, sec_n):
        # a batch = a group of data feeded to the net at one time
        # one epoch is trained for several rounds. The same data is used repeatedly.
        # training.csv's data is splitted into several epochs in use
        self.batch_size = batch_s
        self.ptr = 0 # pointer: index the current line of data
        self.data_file = file 
        self.total_epoch = sec_n  # number of dividing sets, hard coded
        self.total_round = total_epoch

        # split the data into training and validation set
        data_splitter = DataSplitter(file)
        data_splitter.split()

        # configuration
        stu_set = set([])
        exer_set = set([])
        knowledge_set = set([])

        latest_time_map = dict() # map recording the latest test time of each student

        with open(self.data_file, "r", encoding="utf-8-sig") as f: # data_file: whole
            csv_reader = csv.DictReader(f, skipinitialspace=True)
            self.data = list(csv_reader)

        for sample in self.data: # each sample is a piece of data
            stu_set.add(sample['stuUserId'])
            exer_set.add(sample['questionId'])
            kl_ids = json.loads(sample['knowledgeTagIds'])
            for kl in kl_ids: # more than 1 knowledge point possible
                knowledge_set.add(kl)
            latest_time_map[sample['stuUserId']] = sample['startDatetime']

        self.student_n = len(stu_set)
        self.exer_n = len(exer_set)
        self.knowledge_n = len(knowledge_set)
        self.data_len = len(self.data) # length of all data(train + validate)

        # write basic information of the dataset into a txt workfile
        config = str(self.student_n) + ', ' + str(self.exer_n) + ', ' + str(self.knowledge_n) + '\n'
        config += 'student_n, exercise_n, knowledge_n' + '\n'
        config += str(self.data_len) + '\n'
        config += 'Number of samples' + '\n'
        config += self.data_file + '\n'
        config += 'Data file'
        with open("../config/config.txt", "w") as outfile:
            outfile.write(config)
        # sorted list of data
        self.stu_list = sorted(stu_set)
        self.exer_list = sorted(exer_set)
        self.knowledge_list = sorted(knowledge_set)
        # list of the idexes 
        stu_index = list(range(1, self.student_n + 1))
        exer_index = list(range(1, self.exer_n + 1))
        kl_index = list(range(1, self.knowledge_n + 1))
        # 对学生/习题/知识进行编号并储存编号
        self.stu_map = dict(zip(self.stu_list, stu_index))
        self.exer_map = dict(zip(self.exer_list, exer_index))
        self.knowledge_map = dict(zip(self.knowledge_list, kl_index))
        # dump the data as json file
        json_stu = json.dumps(self.stu_map, indent=4)
        json_exer = json.dumps(self.exer_map, indent=4)
        json_kl = json.dumps(self.knowledge_map, indent=4)
        json_latest_time = json.dumps(latest_time_map, indent=4)

        with open("../config/stu_map.json", "w") as outfile:
            outfile.write(json_stu)
        with open("../config/exer_map.json", "w") as outfile:
            outfile.write(json_exer)
        with open("../config/knowledge_map.json", "w") as outfile:
            outfile.write(json_kl)
        with open("../config/stu_latest_time_map.json", "w") as outfile:
            outfile.write(json_latest_time)
  
        self.curr_epoch = -1 #？
        self.curr_round = 1 #？

        # load the data for training after split
        with open('../config/data4training.csv', "r", encoding="utf-8-sig") as f:
            csv_reader = csv.DictReader(f, skipinitialspace=True)
            self.data = list(csv_reader)
        self.data_len = len(self.data)
        # self.data_file: 原始文件（未split）
        # self.data：training.csv

    # read a batch of formalized data
    def next_batch(self):
        rg = self.batch_size
        is_end = False 
        if self.is_section_end():
            # To read some data at each section end when the number of data is smaller than 32,
            # the number 16 can be changed
            rg = 16
            is_end = True
        #  if self.is_end():
        #      return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, question_types = [], [], [], [], []
        for count in range(rg): # in one batch
            log = self.data[self.ptr + count] # current index of data line
            # knowledge embedding: for the current exercise, corresponding knowledge exists: 1; else: 0
            # dimension: knowledge_n
            knowledge_emb = [0.] * self.knowledge_n
            for knowledge_code in json.loads(log['knowledgeTagIds']):
                knowledge_emb[self.knowledge_map[knowledge_code] - 1] = 1.0
            y = float(log['scorePercentage'])
            # 学生编号 -- 方便one-hot
            input_stu_ids.append(self.stu_map[log['stuUserId']] - 1)
            # 习题编号 --方便one-hot
            input_exer_ids.append(self.exer_map[log['questionId']] - 1)
            # 每个习题的knowledge embedding
            input_knowledge_embs.append(knowledge_emb)
            # 分数
            ys.append(y)
            qt = log['questionType']
            # 习题类型
            question_types.append(qt_transformer(qt))

        self.ptr += self.batch_size
        # print(self.ptr)

        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowledge_embs), torch.Tensor(ys), np.array(question_types), is_end

    def is_end(self):
        if self.ptr + self.batch_size > self.data_len:
            return True
        else:
            return False

    def is_epoch_end(self):
        if self.ptr + self.batch_size > self.data_len * (
                self.curr_epoch / float(self.total_epoch)) and self.curr_round == self.total_round:
            self.curr_round = 1
            self.curr_epoch += 1
            return True
        else:
            return False

    # section end: having run all data of the current epoch
    # to do: train a next round
    def is_section_end(self):
        if self.ptr + self.batch_size > self.data_len * (self.curr_epoch / float(self.total_epoch)):
            self.curr_round += 1
            self.ptr = int(self.data_len * ((self.curr_epoch - 1) / float(self.total_epoch)))
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def get_knowledge_dim(self):
        return self.knowledge_n


class ValTestDataLoaderCSV(object):
    def __init__(self, file):
        self.batch_size = 32
        self.data = []
        self.data_file = file # the val/test file

        config_file = '../config/config.txt'
        with open(self.data_file, "r", encoding="utf-8-sig") as f:
            csv_reader = csv.DictReader(f, skipinitialspace=True)
            self.data = list(csv_reader)
        with open(config_file) as i_f:
            _, _, self.knowledge_n = i_f.readline().split(',')
            i_f.readline()
        self.data_len = len(self.data)

        self.knowledge_n = int(self.knowledge_n)

        self.ptr = 0

        # load the mappings generated by train set
        with open('../config/stu_map.json', encoding='utf8') as i_f:
            self.stu_map = json.load(i_f)
        with open('../config/exer_map.json', encoding='utf8') as i_f:
            self.exer_map = json.load(i_f)
        with open('../config/knowledge_map.json', encoding='utf8') as i_f:
            self.knowledge_map = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, question_types = [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_n
            for knowledge_code in json.loads(log['knowledgeTagIds']):
                knowledge_emb[self.knowledge_map[knowledge_code] - 1] = 1.0
            y = float(log['scorePercentage'])
            input_stu_ids.append(self.stu_map[log['stuUserId']] - 1)
            input_exer_ids.append(self.exer_map[log['questionId']] - 1)
            input_knowledge_embs.append(knowledge_emb)
            ys.append(y)
            question_types.append(qt_transformer(log['questionType']))

        self.ptr += self.batch_size

        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowledge_embs), torch.Tensor(ys), question_types

    def is_end(self):
        if self.ptr + self.batch_size > self.data_len:
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def get_knowledge_dim(self):
        return self.knowledge_n

## split the data into training and validating(last month)
## place the splited data into a newly created folder
class DataSplitter(object):
    def __init__(self, whole_file):
        self.wholeFile = whole_file # the file to split
        self.trainFile = '../config/data4training.csv' 
        self.valFile = '../config/data4validation.csv'

    def split(self):
        # if not such directory, create one
        if not os.path.exists('../config/'): 
            os.makedirs('../config/')

        with open(self.wholeFile, "r", encoding="utf-8-sig") as f:
            csv_reader = csv.DictReader(f, skipinitialspace=True) # read in data file
            whole_data = list(csv_reader) # transform data into list
        last_index = len(whole_data) - 1
        last_date = whole_data[last_index]['startDatetime'] # startDateTime: 开始测试时间
        last_month = last_date.split('/')[0]
        # split all data of the last month as validation data
        val_data = []
        while whole_data[-1]['startDatetime'].split('/')[0] == last_month:
            val_data.append(whole_data.pop())
        val_data.reverse()
        # write training and validation data separately into two files
        keys = whole_data[0].keys()
        with open(self.trainFile, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(whole_data)
        with open(self.valFile, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(val_data)
