import torch
import numpy as np
import json
import sys
from data_loader import ValTestDataLoader
from model import Net


# can be changed according to config.txt
exer_n = 17746
knowledge_n = 123
student_n = 4163


def test(epoch):
    data_loader = ValTestDataLoader('test')
    net = Net(student_n, exer_n, knowledge_n)
    device = torch.device('cpu')
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, '../model/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
        out_put = out_put.view(-1)

        pred_all += out_put.tolist()
        label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    print('epoch= %d, rmse= %f' % (epoch, rmse))
    with open('../result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, rmse= %f\n' % (epoch, rmse))


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def get_status(epoch):
    '''
    An example of getting student's knowledge status
    :return:
    '''
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
    if (len(sys.argv) != 2) or (not sys.argv[1].isdigit()):
        print('command:\n\tpython predict.py {epoch}\nexample:\n\tpython predict.py 70')
        exit(1)

    # global student_n, exer_n, knowledge_n
    with open('../config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    test(int(sys.argv[1]))
    get_status(int(sys.argv[1]))
    get_exer_params(int(sys.argv[1]))

