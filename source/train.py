import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net


# can be changed according to config.txt
exer_n = 17746
knowledge_n = 123
student_n = 4163
# can be changed according to command parameter
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5


def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')

    loss_function = nn.MSELoss()
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)

            # grad_penalty = 0
            loss = loss_function(torch.squeeze(output, 1), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # validate and save current model every epoch
        rmse = validate(net, epoch)
        save_snapshot(net, '../model/model_epoch' + str(epoch + 1))


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)

    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    print('epoch= %d, rmse= %f' % (epoch + 1, rmse))
    with open('../result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, rmse= %f\n' % (epoch + 1, rmse))

    return rmse


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    # global student_n, exer_n, knowledge_n, device
    with open('../config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    train()
