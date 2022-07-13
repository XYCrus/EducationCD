import json
import torch


class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, file):
        self.batch_size = 32
        self.ptr = 0
        self.data_file = file

        #configuration
        stu_set = set([])
        exer_set = set([])
        knowledge_set = set([])

        with open(self.data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        
        for sample in self.data:
            stu_set.add(sample['user_id'])
            exer_set.add(sample['exer_id'])
            for kl in sample['knowledge_code']:
                knowledge_set.add(kl)

        #Load validation data to map
        with open('../data/val_set_transformed.json', encoding='utf8') as i_f:
            valData = json.load(i_f)

        for stu in valData:
            #stu_set.add(stu['user_id'])
            for log in stu['logs']:
                exer_set.add(log['exer_id'])

        with open('../data/test_set_transformed.json', encoding='utf8') as i_f:
            testData = json.load(i_f)

        for stu in testData:
            #stu_set.add(stu['user_id'])
            for log in stu['logs']:
                exer_set.add(log['exer_id'])


        self.student_n = len(stu_set)
        self.exer_n = len(exer_set)
        self.knowledge_n = len(knowledge_set)
        self.data_len = len(self.data)

        config = str(self.student_n) + ', ' + str(self.exer_n) + ', ' + str(self.knowledge_n) + '\n'
        config += 'student_n, exercise_n, knowledge_n' + '\n'
        config += str(self.data_len) + '\n'
        config += 'Number of samples' + '\n'
        config += self.data_file + '\n'
        config += 'Train file'
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

        with open("../config/stu_map.json", "w") as outfile:
            outfile.write(json_stu)
        with open("../config/exer_map.json", "w") as outfile:
            outfile.write(json_exer)
        with open("../config/knowledge_map.json", "w") as outfile:
            outfile.write(json_kl)
  
        self.knowledge_dim = self.knowledge_n



    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[self.knowledge_map[knowledge_code] - 1] = 1.0
            y = log['score']
            input_stu_ids.append(self.stu_map[log['user_id']] - 1)
            input_exer_ids.append(self.exer_map[log['exer_id']] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size

        
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.Tensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def get_knowledge_dim(self):
        return self.knowledge_n


class ValTestDataLoader(object):
    def __init__(self, file, d_type='validation'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type
        
        if d_type == 'validation':
            self.data_file = '../data/val_set_transformed.json'
        else:
            self.data_file = file
        config_file = '../config/config.txt'
        with open(self.data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            #i_f.readline()
            _, _, self.knowledge_n = i_f.readline().split(',')
        self.knowledge_n = int(self.knowledge_n)

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
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(self.stu_map[str(user_id)] - 1)
            input_exer_ids.append(self.exer_map[str(log['exer_id'])] - 1)
            knowledge_emb = [0.] * self.knowledge_n
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[self.knowledge_map[str(knowledge_code)] - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        #print(input_stu_ids)
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.Tensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
