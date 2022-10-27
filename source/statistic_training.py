import string
import sys
import os.path
import pandas as pd
import numpy as np
import json

def get_summary(df, stu_id):
    sub_df = df[df['stuUserId'] == stu_id]
    scores = {}
    count_1 = {}
    count_0 = {}
    single_apr = {}
    complex_apr = {}
    for i in range(sub_df.shape[0]):
        score = sub_df['scorePercentage'].iloc[i]
        klgs = json.loads(sub_df['knowledgeTagIds'].iloc[i])
        for klg in klgs:
            if klg not in scores.keys():
                scores[klg] = []
            if klg not in count_1.keys():
                count_1[klg] = 0
            if klg not in count_0.keys():
                count_0[klg] = 0
            if klg not in single_apr.keys():
                single_apr[klg] = 0
            if klg not in complex_apr.keys():
                complex_apr[klg] = 0

            scores[klg].append(score)
            if score == 1: 
                count_1[klg] += 1
            elif score == 0:
                count_0[klg] += 1
            if len(klgs) == 1:
                single_apr[klg] += 1
            else:
                complex_apr[klg] += 1

    avg_scores = {}
    for klg in scores.keys():
        avg_scores[klg] = np.mean(np.array(scores[klg]))
    return avg_scores, count_0, count_1, single_apr, complex_apr

def extract_stu_klg(model):
    # save as csv file
    stu_klg = pd.DataFrame(model, columns=['knowledge_ids', 'score_rate'])
    stu_klg.to_csv("../result/student_knowledge.csv")

def extract_stu_klg_json(json_data):
    with open("../result/student_knowledge_sta.json", "w") as outfile:
        json.dump(json_data, outfile, indent=4)

def extract_stu_avg(model: pd.DataFrame):
    grouped = model['score_rate'].groupby('stu_user_id').mean()
    model['stu_avg'] = grouped 
    stu_avg_dict = dict(grouped)
    with open("../result/stu_avg.json", "w") as outfile:
        json.dump(stu_avg_dict, outfile, indent=4)

def extract_overall_avg(model: pd.DataFrame):
    overall_avg = np.mean(model['score_rate'])
    model['overall_avg'] = overall_avg
    with open("../result/overall_avg.json", "w") as outfile:
        outfile.write(json.dumps(overall_avg))

def create_statistic_model(datapath: string):
    # load data
    df = pd.read_csv(datapath)

    # collect student - knowledge pairs
    stu_col = []
    klg_col = []
    pair_set = set([])
    for i in range(df.shape[0]):
        stu = df['stuUserId'].iloc[i]
        knowledges = json.loads(df['knowledgeTagIds'].iloc[i])
        for klg in knowledges:
            pair_set.add((stu,klg))
    for pair in pair_set:
        stu_col.append(pair[0])
        klg_col.append(pair[1])

    # build new dataframe
    data = {'stu_user_id': np.array(stu_col), 'knowledge_ids': np.array(klg_col)}
    model = pd.DataFrame(data=data)

    # build remaining column and json data

    json_data = []

    ## initialization
    stu = model['stu_user_id'].iloc[0]
    summary = get_summary(df, stu)
    klg = model['knowledge_ids'].iloc[0]

    ### record info into cols
    avg_score_col = [summary[0][klg]]
    col_0 = [summary[1][klg]]
    col_1 = [summary[2][klg]]
    col_sing = [summary[3][klg]]
    col_comp = [summary[4][klg]]

    ### record info into json data
    current_student_scores = {"stuUserId": str(stu)}
    knowledgScores = []
    avg_scores = summary[0]
    for (key, val) in avg_scores.items():
        knowledgScores.append({"knowledgeTagId": key, "score": val})
    current_student_scores["knowledgScores"] = knowledgScores
    json_data.append(current_student_scores)

    ## loop
    for i in range(1, model.shape[0]):
        stu = model['stu_user_id'].iloc[i]

        if stu != model['stu_user_id'].iloc[i-1]: # need to change summary
            summary = get_summary(df, stu)
            ### record json data
            current_student_scores = {"stuUserId": str(stu)}
            knowledgScores = []
            avg_scores = summary[0]
            for (key, val) in avg_scores.items():
                knowledgScores.append({"knowledgeTagId": key, "score": val})
            current_student_scores["knowledgScores"] = knowledgScores
            json_data.append(current_student_scores)

        klg = model['knowledge_ids'].iloc[i]
        avg_score_col.append(summary[0][klg])
        col_0.append(summary[1][klg])
        col_1.append(summary[2][klg])
        col_sing.append(summary[3][klg])
        col_comp.append(summary[4][klg])

    ## build columns
    model['score_rate'] = np.array(avg_score_col)
    model['1_score_count'] = np.array(col_1)
    model['0_score_count'] = np.array(col_0)
    model['single_knowledge'] = np.array(col_sing)
    model['multiple_knowledge'] = np.array(col_comp)
    model['total_count'] = model['single_knowledge'] + model['multiple_knowledge']

    model = model.sort_values(by = ['stu_user_id', 'knowledge_ids'], ascending=(True, True))
    model = model.set_index("stu_user_id", drop=True)
    for i in range(model.shape[0]):
        model['knowledge_ids'].iloc[i] = json.dumps([model['knowledge_ids'].iloc[i]])

    # generate student_knowledge file, csv & json
    extract_stu_klg(model)
    extract_stu_klg_json(json_data)
    # calculate per student average, write into json
    extract_stu_avg(model)
    # calculate overall average, write into json
    extract_overall_avg(model)

    # save the dataframe to file
    model.to_csv("../model/model.csv")

def check_folder():
    if not os.path.exists('../config'):
        os.mkdir('../config')
    if not os.path.exists('../model'):
        os.mkdir('../model')
    if not os.path.exists('../result'):
        os.mkdir('../result')


if __name__ == '__main__':
    # sys.argv catches the command parameters typed (sep with ' ')
    # read in and record parameters

    # the first component of the string typed
    data_file = sys.argv[1]
    if not data_file.endswith('.csv'):
        print('wrong file type')
        exit(1)

    check_folder()
    create_statistic_model(data_file)
