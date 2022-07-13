import torch
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import ValTestDataLoader
from model import Net
from fake_test_generator import FakeTestGenerator
import csv

#Can be updated
test_file = '../data/test_set_transformed.json'

def test(epoch):
    data_loader = ValTestDataLoader(test_file, 'test')
    with open('../config/config.txt') as configFile:
        student_n, exer_n, knowledge_n = configFile.readline().split(',')
    student_n, exer_n, knowledge_n = int(student_n), int(exer_n), int(knowledge_n)

    net = Net(student_n, exer_n, knowledge_n)
    device = torch.device('cpu')
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, '../model/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    #i = 0
    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
        out_put = out_put.view(-1)

        # compute accuracy
        for i in range(len(labels)):
            if (abs(labels[i] - out_put[i]) < 0.1):
                correct_count += 1
        exer_count += len(labels)
        pred_all += out_put.tolist()
        label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    #auc = roc_auc_score(label_all, pred_all)
    print('epoch= %s, accuracy= %f, rmse= %f' % (str(epoch), accuracy, rmse))
    with open('../result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %s, accuracy= %f, rmse= %f' % (str(epoch), accuracy, rmse))


def test_default(epoch, haveInputFile):
    with open('../config/config.txt') as configFile:
        student_n, exer_n, knowledge_n = configFile.readline().split(',')
    student_n, exer_n, knowledge_n = int(student_n), int(exer_n), int(knowledge_n)
    print(student_n, exer_n, knowledge_n)

    net = Net(student_n, exer_n, knowledge_n)
    device = torch.device('cpu')
    print('testing model...')
    load_snapshot(net, '../model/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    data = []

    with open('../config/stu_map.json', encoding='utf8') as i_f:
        stu_map = json.load(i_f)
    with open('../config/exer_map.json', encoding='utf8') as i_f:
        exer_map = json.load(i_f)
    with open('../config/knowledge_map.json', encoding='utf8') as i_f:
        knowledge_map = json.load(i_f)

    if haveInputFile:
        data = FakeTestGenerator.generate(test_file)
    else:
        data = FakeTestGenerator.generate("../data/train_set_transformed.json")
    data_len = len(data)

    #Load Data
    input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
    for count in range(data_len):
        log = data[count]
        knowledge_emb = [0.] * knowledge_n
        for knowledge_code in log['knowledge_code']:
            knowledge_emb[knowledge_map[str(knowledge_code)] - 1] = 1.0
        y = log['score']
        input_stu_ids.append(stu_map[str(log['user_id'])] - 1)
        input_exer_ids.append(log['exer_id'])
        input_knowedge_embs.append(knowledge_emb)
        ys.append(y)
    
    input_stu_ids, input_exer_ids, input_knowledge_embs, ys = torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.Tensor(ys)

    output = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
    output = output.view(-1).tolist()
    # print(len(output))
    for i in range(data_len):
        data[i]['score'] = output[i]
    
    data = json.dumps(data, indent = 4)
    with open("../result/student_knowledge.json", "w") as outfile:
        outfile.write(data)


def test_csv(epoch):
    with open('../config/config.txt') as configFile:
        student_n, exer_n, knowledge_n = configFile.readline().split(',')
    student_n, exer_n, knowledge_n = int(student_n), int(exer_n), int(knowledge_n)

    net = Net(student_n, exer_n, knowledge_n)
    device = torch.device('cpu')
    print('testing model...')
    load_snapshot(net, '../model/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    data = []

    with open('../config/stu_map.json', encoding='utf8') as i_f:
        stu_map = json.load(i_f)
    with open('../config/exer_map.json', encoding='utf8') as i_f:
        exer_map = json.load(i_f)
    with open('../config/knowledge_map.json', encoding='utf8') as i_f:
        knowledge_map = json.load(i_f)

    data = FakeTestGenerator.generate(test_file)
    data_len = len(data)

    #Load Data
    input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
    for count in range(data_len):
        log = data[count]
        knowledge_emb = [0.] * knowledge_n
        for knowledge_id in log['knowledge_ids']:
            knowledge_emb[knowledge_map[knowledge_id] - 1] = 1.0
        y = log['score_percentage']
        input_stu_ids.append(stu_map[log['stu_user_id']] - 1)
        input_exer_ids.append(log['question_id'] - 1)
        input_knowedge_embs.append(knowledge_emb)
        ys.append(y)
    
    input_stu_ids, input_exer_ids, input_knowledge_embs, ys = torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.Tensor(ys)

    output = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
    output = output.view(-1).tolist()
    # print(len(output))
    for i in range(data_len):
        data[i]['score_percentage'] = output[i]

    keys = data[0].keys()
    with open('../result/student_knowledge.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    

def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def get_status(epoch):
    '''
    An example of getting student's knowledge status
    :return:
    '''
    with open('../config/config.txt') as configFile:
        student_n, exer_n, knowledge_n = configFile.readline().split(',')
    student_n, exer_n, knowledge_n = int(student_n), int(exer_n), int(knowledge_n)

    net = Net(student_n, exer_n, knowledge_n)
    load_snapshot(net, '../model/model_epoch' + str(epoch))
    with open('../result/student_stat.txt', 'w', encoding='utf8') as output_file:
        for stu_id in range(student_n):
            # get knowledge status of student with stu_id (index)
            status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
            output_file.write(str(status) + '\n')

def get_exer_params(epoch):
    '''
    An example of getting exercise's parameters (knowledge difficulty and exercise discrimination)
    :return:
    '''
    with open('../config/config.txt') as configFile:
        student_n, exer_n, knowledge_n = configFile.readline().split(',')
    student_n, exer_n, knowledge_n = int(student_n), int(exer_n), int(knowledge_n)

    net = Net(student_n, exer_n, knowledge_n)
    load_snapshot(net, '../model/model_epoch' + str(epoch))  # load model
    exer_params_dict = {}
    for exer_id in range(exer_n):
        # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
        k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id]))
        exer_params_dict[exer_id + 1] = (k_difficulty.tolist()[0], e_discrimination.tolist()[0])
    with open('../result/exer_params.txt', 'w', encoding='utf8') as o_f:
        o_f.write(str(exer_params_dict))


if __name__ == '__main__':

    with open('../config/config.txt', 'r') as configFile:
        for i in range(4):
            configFile.readline()
        test_file = str(configFile.readline())[:-1]
    print("predicting: " + test_file)

    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        if test_file.endswith('.json'):
            test_default(sys.argv[1], False)
        elif test_file.endswith('.csv'):
            test_csv(sys.argv[1])
        

    elif len(sys.argv) == 1:
        if test_file.endswith('.json'):
            test_default('_latest', False)
        elif test_file.endswith('.csv'):
            test_csv('_latest')
        # get_status('_latest')
        # get_exer_params('_latest')

    elif (len(sys.argv) == 2) and (not sys.argv[1].isdigit()):
        print('b')
        test_file = sys.argv[1]
        test('_latest', True)
        get_status('_latest')
        get_exer_params('_latest')
    

    else:
        print('command:\n\tpython predict.py {epoch}\nexample:\n\tpython predict.py 70')
        exit(1)