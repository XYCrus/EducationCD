import json
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from get_average import GetAverage
from data_loader import TrainDataLoader, ValTestDataLoader
from fake_test_generator import FakeTestGenerator
from model import Net
from data_loader_csv import TrainDataLoaderCSV
from data_loader_csv import ValTestDataLoaderCSV
from predict import test_csv

<<<<<<< source/train.py

device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
=======
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
>>>>>>> source/train.py
epoch_n = 5
train_file = '../data/train_set_transformed.json'


def train():
    if train_file.endswith('.json'):
        data_loader = TrainDataLoader(train_file)
    else:
        data_loader = TrainDataLoaderCSV(train_file, epoch_n)
    exer_n = data_loader.exer_n
    knowledge_n = data_loader.knowledge_n
    student_n = data_loader.student_n
    net = Net(student_n, exer_n, knowledge_n)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')
    # loss_function = nn.MSELoss()
    # The score for evaluating a model(epoch), best_performance = accuracy - rmse (Can be changed for better training)
    best_performance = -10.0

    data = FakeTestGenerator.generate(train_file)

    for epoch in range(3):
        # data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        data_loader.curr_epoch = epoch + 1
        while not data_loader.is_epoch_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, target, question_types, section_end_flag \
                = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, target = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), target.to(device)
            optimizer.zero_grad()
            output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output = torch.squeeze(output, 1)

            loss = calculate_loss(output, target, question_types)
            # loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 5000 == 4999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 5000))
                running_loss = 0.0

            if section_end_flag:
                # validate and save current model every epoch
                validate(net, epoch, data_loader.curr_round - 1)
                score_var = validate_average_deviation(net, data)
                # save_snapshot(net, '../model/model_epoch' + str(epoch + 1))
                if data_loader.curr_round > 20 and score_var > best_performance:
                    best_performance = score_var
                    save_snapshot(net, '../model/model_epoch_latest')

        validate(net, epoch, epoch_n)
        score_var = validate_average_deviation(net, data)
        # save_snapshot(net, '../model/model_epoch' + str(epoch + 1))
        if score_var > best_performance:
            best_performance = score_var
            save_snapshot(net, '../model/model_epoch_latest')


def validate(model, section_number, round_number):
    if train_file.endswith('.json'):
        data_loader = ValTestDataLoader('validation')
    else:
        # data_loader = ValTestDataLoaderCSV(train_file)
        data_loader = ValTestDataLoaderCSV('../config/data4validation.csv')

    with open('../config/config.txt') as configFile:
        student_n, exer_n, knowledge_n = configFile.readline().split(',')

    student_n, exer_n, knowledge_n = int(student_n), int(exer_n), int(knowledge_n)
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    # data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all, question_types_all = [], [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, question_types = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if question_types[i] == 4:
                # error acceptable
                if abs(labels[i] - output[i]) < 0.1:
                    correct_count += 1
            elif question_types[i] == 1 or question_types[i] == 2:
                if labels[i] == 1 and output[i] > 0.8:
                    correct_count += 1
                elif labels[i] == 0.6 and 0.5 < output[i] <= 0.8:
                    correct_count += 1
                elif labels[i] == 0.4 and 0.2 < output[i] <= 0.5:
                    correct_count += 1
                elif labels[i] == 0 and output[i] <= 0.2:
                    correct_count += 1
            elif question_types[i] == 3:
                if labels[i] == 1 and output[i] > 0.5:
                    correct_count += 1
                elif labels[i] == 0 and output[i] <= 0.5:
                    correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()
        question_types_all.extend(question_types)

    pred_all = np.array(pred_all)
    for i in range(len(question_types_all)):
        # Mapping
        if question_types_all[i] == 3:
            pred_all[i] = 1 if pred_all[i] > 0.5 else 0
        elif question_types_all[i] == 1 or question_types_all[i] == 2:
            if pred_all[i] > 0.8:
                pred_all[i] = 1
            elif pred_all[i] > 0.5:
                pred_all[i] = 0.6
            elif pred_all[i] > 0.2:
                pred_all[i] = 0.4
            else:
                pred_all[i] = 0
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # r2 = r2_score(label_all, pred_all)
    print('section= %d, epoch= %d, accuracy= %f, rmse= %f\n' % (
         section_number + 1, round_number, accuracy, rmse))
    with open('../result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('section= %d, epoch= %d, accuracy= %f, rmse= %f\n' % (
            section_number + 1, round_number, accuracy, rmse))

    return rmse, accuracy


def validate_average_deviation(model, data):
    with open('../config/config.txt') as configFile:
        student_n, exer_n, knowledge_n = configFile.readline().split(',')
    student_n, exer_n, knowledge_n = int(student_n), int(exer_n), int(knowledge_n)
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    # data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(torch.device('cpu'))
    net.eval()

    with open('../config/stu_map.json', encoding='utf8') as i_f:
        stu_map = json.load(i_f)
    with open('../config/knowledge_map.json', encoding='utf8') as i_f:
        knowledge_map = json.load(i_f)
    with open('../config/stu_latest_time_map.json', encoding='utf8') as i_f:
        stu_latest_time_map = json.load(i_f)

    data_len = len(data)

    # Load Data
    input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
    for count in range(data_len):
        log = data[count]
        knowledge_emb = [0.] * knowledge_n
        for knowledge_id in log['knowledgeTagIds']:
            knowledge_emb[knowledge_map[knowledge_id] - 1] = 1.0
        y = log['scorePercentage']
        input_stu_ids.append(stu_map[log['stuUserId']] - 1)
        input_exer_ids.append(log['questionId'] - 1)
        input_knowedge_embs.append(knowledge_emb)
        ys.append(y)

    input_stu_ids, input_exer_ids, input_knowledge_embs, ys = torch.LongTensor(input_stu_ids), torch.LongTensor(
        input_exer_ids), torch.Tensor(input_knowedge_embs), torch.Tensor(ys)

    output = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
    output = output.view(-1).tolist()
    net.train()
    current_student_id = ""
    current_student_scores = dict()
    total_score = 0
    average_data = []

    for i in range(data_len):
        data[i]['scorePercentage'] = output[i]
        if data[i]['stuUserId'] != current_student_id:
            if 'knowledgeScores' in current_student_scores:
                average_score = total_score / len(current_student_scores['knowledgeScores'])
                average_data.append(average_score)
            current_student_scores = dict()
            total_score = 0
            current_student_id = data[i]['stuUserId']
            current_student_scores['stuUserId'] = data[i]['stuUserId']
            current_student_scores['startDatetime'] = stu_latest_time_map[data[i]['stuUserId']]
            current_student_scores['knowledgeScores'] = [{
                'knowledgeTagId': data[i]['knowledgeTagIds'][0],
                'score': output[i]
            }]
        else:
            current_student_scores['knowledgeScores'].append({
                'knowledgeTagId': data[i]['knowledgeTagIds'][0],
                'score': output[i]
            })
            total_score += output[i]
    if 'knowledgeScores' in current_student_scores:
        average_score = total_score / len(current_student_scores['knowledgeScores'])
        average_data.append(average_score)

    score_var = np.var(average_data)
    return score_var


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def calculate_loss(output, target, question_types):
    for i in range(len(question_types)):
        # Mapping
        if question_types[i] == 3:
            if output[i] > 0.5 and target[i] == 1:
                target[i] = output[i]
            elif output[i] <= 0.5 and target[i] == 0:
                target[i] = output[i]
        if question_types[i] == 1 or question_types[i] == 2:
            if output[i] > 0.8 and target[i] == 1:
                target[i] = output[i]
            elif 0.5 < output[i] <= 0.8 and target[i] == 0.6:
                target[i] = output[i]
            elif 0.2 < output[i] <= 0.5 and target[i] == 0.4:
                target[i] = output[i]
            elif output[i] < 0.2 and target[i] == 0:
                target[i] = output[i]
    mse = nn.MSELoss()
    # l1 = nn.L1Loss()
    return mse(output, target)


def check_folder():
    if not os.path.exists('../config'):
        os.mkdir('../config')
    if not os.path.exists('../model'):
        os.mkdir('../model')
    if not os.path.exists('../result'):
        os.mkdir('../result')


if __name__ == '__main__':
    if (len(sys.argv) != 4) or ((sys.argv[2] != 'cpu') and ('cuda:' not in sys.argv[2])) or (not sys.argv[3].isdigit()):
        print('command:\n\tpython train.py {file} {device} {epoch}\nexample:\n\tpython train.py xxx.json cuda:0 70')
        exit(1)
    else:
        train_file = sys.argv[1]
        if not train_file.endswith('.csv') and not train_file.endswith('.json'):
            print('wrong file type')
            exit(1)
        device = torch.device(sys.argv[2])
        epoch_n = int(sys.argv[3])
        epoch_n = max(20, epoch_n)

    check_folder()
    train()

    print("predicting: " + train_file)
    test_csv('_latest', train_file)
    GetAverage.getAverage()
