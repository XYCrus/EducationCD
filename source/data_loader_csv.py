import json
import torch
import csv
import numpy as np
import os


class TrainDataLoaderCSV(object):
    '''
    data loader for training
    '''
    def __init__(self, file, total_epoch):
        self.batch_size = 1
        self.ptr = 0
        self.data_file = file
        self.total_epoch = total_epoch
        self.total_round = total_epoch

        dataSplitter = DataSplitter(file)
        dataSplitter.split()

        #configuration
        stu_set = set([])
        exer_set = set([])
        knowledge_set = set([])

        latest_time_map = dict()

        with open(self.data_file, "r", encoding="utf-8-sig") as f:
            csv_reader = csv.DictReader(f, skipinitialspace=True)
            self.data = list(csv_reader)
        
        for sample in self.data:
            stu_set.add(sample['stuUserId'])
            exer_set.add(sample['questionId'])
            kl_ids = json.loads(sample['knowledgeTagIds'])
            for kl in kl_ids:
                knowledge_set.add(kl)
            latest_time_map[sample['stuUserId']] = sample['startDatetime']

        self.student_n = len(stu_set)
        self.exer_n = len(exer_set)
        self.knowledge_n = len(knowledge_set)
        self.data_len = len(self.data)

        config = str(self.student_n) + ', ' + str(self.exer_n) + ', ' + str(self.knowledge_n) + '\n'
        config += 'student_n, exercise_n, knowledge_n' + '\n'
        config += str(self.data_len) + '\n'
        config += 'Number of samples' + '\n'
        config += self.data_file + '\n'
        config += 'Data file'
        with open("../config/config.txt", "w") as outfile:
            outfile.write(config)

        self.stu_list = sorted(stu_set)
        self.exer_list = sorted(exer_set)
        self.knowledge_list = sorted(knowledge_set)

        stu_index = list(range(1, self.student_n + 1))
        exer_index = list(range(1, self.exer_n + 1))
        kl_index = list(range(1, self.knowledge_n + 1))

        self.stu_map = dict(zip(self.stu_list, stu_index))
        self.exer_map = dict(zip(self.exer_list, exer_index))
        self.knowledge_map = dict(zip(self.knowledge_list, kl_index))

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
  
        self.curr_epoch = -1
        self.curr_round = 1

        # load the data for training after split
        with open('../config/data4training.csv', "r", encoding="utf-8-sig") as f:
            csv_reader = csv.DictReader(f, skipinitialspace=True)
            self.data = list(csv_reader)
        self.data_len = len(self.data)

    def next_batch(self):
        rg = self.batch_size
        if self.is_section_end():
            #To read some data at each section end when the number of data is smaller than 32, the number 16 can be changed
            rg = 16
        # if self.is_end():
        #     return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys, questionTypes = [], [], [], [], []
        for count in range(rg):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_n
            for knowledge_code in json.loads(log['knowledgeTagIds']):
                knowledge_emb[self.knowledge_map[knowledge_code] - 1] = 1.0
            y = float(log['scorePercentage'])
            input_stu_ids.append(self.stu_map[log['stuUserId']] - 1)
            input_exer_ids.append(self.exer_map[log['questionId']] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)
            qt = log['questionType']
            questionTypes.append(self.QTTransformer(qt))

        self.ptr += self.batch_size
        #print(self.ptr)
        
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.Tensor(ys), np.array(questionTypes)

    def is_end(self):
        if self.ptr + self.batch_size > self.data_len:
            return True
        else:
            return False

    def is_epoch_end(self):
        if self.ptr + self.batch_size > self.data_len * (self.curr_epoch / float(self.total_epoch)) and self.curr_round == self.total_round:
            self.curr_round = 1
            self.curr_epoch += 1
            return True
        else:
            return False

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

    def QTTransformer(self, qt):
        if qt == 'FILLBLANK':
            return 1
        elif qt == 'MCHOICE':
            return 2
        elif qt == 'SCHOICE':
            return 3
        elif qt == 'SHORTANSWER':
            return 4


class ValTestDataLoaderCSV(object):
    def __init__(self, file):
        self.batch_size = 32
        self.data = []
        self.data_file = file
        
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

        #load the mappings generated by train set
        with open('../config/stu_map.json', encoding='utf8') as i_f:
            self.stu_map = json.load(i_f)
        with open('../config/exer_map.json', encoding='utf8') as i_f:
            self.exer_map = json.load(i_f)
        with open('../config/knowledge_map.json', encoding='utf8') as i_f:
            self.knowledge_map = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys, questionTypes = [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_n
            for knowledge_code in json.loads(log['knowledgeTagIds']):
                knowledge_emb[self.knowledge_map[knowledge_code] - 1] = 1.0
            y = float(log['scorePercentage'])
            input_stu_ids.append(self.stu_map[log['stuUserId']] - 1)
            input_exer_ids.append(self.exer_map[log['questionId']] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)
            questionTypes.append(self.QTTransformer(log['questionType']))

        self.ptr += self.batch_size
        
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.Tensor(ys), questionTypes

    def is_end(self):
        if self.ptr + self.batch_size > self.data_len:
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def get_knowledge_dim(self):
        return self.knowledge_n

    def QTTransformer(self, qt):
        if qt == 'FILLBLANK':
            return 1
        elif qt == 'MCHOICE':
            return 2
        elif qt == 'SCHOICE':
            return 3
        elif qt == 'SHORTANSWER':
            return 4


class DataSplitter(object):
    def __init__(self, wholeFile):
        self.wholeFile = wholeFile
        self.trainFile = '../config/data4training.csv'
        self.valFile = '../config/data4validation.csv'
    
    def split(self):
        if not os.path.exists('../config/'):
            os.makedirs('../config/')
        with open(self.wholeFile, "r", encoding="utf-8-sig") as f:
            csv_reader = csv.DictReader(f, skipinitialspace=True)
            wholeData = list(csv_reader)
        lastIndex = len(wholeData) - 1
        lastDate = wholeData[lastIndex]['startDatetime']
        lastMonth = lastDate.split('/')[0]
        valData = []
        while wholeData[-1]['startDatetime'].split('/')[0] == lastMonth:
            valData.append(wholeData.pop())
        valData.reverse()

        keys = wholeData[0].keys()
        with open(self.trainFile, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(wholeData)
        with open(self.valFile, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(valData)
