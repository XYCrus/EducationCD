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
    stu_klg = pd.DataFrame(model, columns=['stu_user_id','knowledge_ids', 'score_rate'])
    stu_klg = stu_klg.set_index("stu_user_id", drop=True)
    stu_klg.to_csv("../result/student_knowledge_statistics.csv")


def extract_stu_klg_json(json_data):
    json_data = list(json_data.values())
    with open("../result/student_knowledge_statistics.json", "w") as outfile:
        json.dump(json_data, outfile, indent=4)


def extract_klg_avg(model: pd.DataFrame, folder = '../model'):
    df = model.copy()
    df = df.set_index('knowledge_ids',drop=False)
    grouped = df['score_rate'].groupby('knowledge_ids').mean()
    model['klg_avg'] = model['knowledge_ids'].apply(lambda x: grouped.loc[x])
    klg_avg_dict = dict(grouped)
    
    # klg1_score = {}
    # klg1_count = {}
    # klg0_score = {}
    # klg0_count = {}

    # df1 = model[model['flag'] == 1]
    # for i in range(df1.shape[0]):
    #     klg = df1['knowledge_ids'].iloc[i]
    #     score = df1['score_rate'].iloc[i]
    #     count = df1['total_count'].iloc[i]
    #     klg1_score[klg] = klg1_score.get(klg,0) + score*count
    #     klg1_count[klg] = klg1_score.get(klg,0) + count
    # for key in klg1_count.keys():
    #     klg1_score[key] = klg1_score[key] / klg1_count[key]

    # df0 = model[model['flag'] == 0]
    # for i in range(df0.shape[0]):
    #     klg = df0['knowledge_ids'].iloc[i]
    #     if klg not in klg1_count.keys():
    #         score = df0['score_rate'].iloc[i]
    #         count = df0['total_count'].iloc[i]
    #         klg0_score[klg] = klg0_score.get(klg,0) + score*count
    #         klg0_count[klg] = klg0_score.get(klg,0) + count
    # for key in klg0_count.keys():
    #     klg0_score[key] = klg0_score[key] / klg0_count[key]
    
    # klg_avg_dict = dict(klg1_score, **klg0_score)
    # model['klg_avg'] = model['knowledge_ids'].apply(lambda x: klg_avg_dict[x])

    with open("../result/knowledge_average.json", "w") as outfile:
        json.dump(klg_avg_dict, outfile, indent=4)
    

    with open(folder+"/knowledge_average.json", "w") as outfile:
        json.dump(klg_avg_dict, outfile, indent=4)

    return model


def extract_stu_avg(model: pd.DataFrame, folder = '../model'):
    df = model.copy()
    df = df.set_index('stu_user_id',drop=False)
    grouped = df['score_rate'].groupby('stu_user_id').mean()
    model['stu_avg'] = model['stu_user_id'].apply(lambda x: grouped.loc[x])
    stu_score = dict(grouped)
    
    # stu_score = {}
    # stu_count = {}
    # stu_flag = {}

    # df1 = model[model['flag'] == 1]
    # for i in range(df1.shape[0]):
    #     stu = int(df1['stu_user_id'].iloc[i])
    #     score = df1['score_rate'].iloc[i]
    #     count = df1['total_count'].iloc[i]
    #     stu_score[stu] = stu_score.get(stu,0) + score*count
    #     stu_count[stu] = stu_score.get(stu,0) + count
    #     stu_flag[stu] = 1

    # df0 = model[model['flag'] == 0]
    # for i in range(df0.shape[0]):
    #     stu = int(df0['stu_user_id'].iloc[i])
    #     if not stu_flag.get(stu,0):
    #         score = df0['score_rate'].iloc[i]
    #         count = df0['total_count'].iloc[i]
    #         stu_score[stu] = stu_score.get(stu,0) + score*count
    #         stu_count[stu] = stu_score.get(stu,0) + count
    # for key in stu_count.keys():
    #     stu_score[key] = stu_score[key] / stu_count[key]
    
    # model['stu_avg'] = model['stu_user_id'].apply(lambda x: stu_score[x])
    
    with open("../result/student_average.json", "w") as outfile:
        json.dump(stu_score, outfile, indent=4)
        
    with open(folder+"/student_average.json", "w") as outfile:
        json.dump(stu_score, outfile, indent=4)

    return model


def collect_pairs(df):
    pair_set = set([])
    for i in range(df.shape[0]):
        stu = df['stuUserId'].iloc[i]
        knowledges = json.loads(df['knowledgeTagIds'].iloc[i])
        for klg in knowledges:
            pair_set.add((stu,klg))
    return pair_set


def create_statistic_model(datapath: string, folder = '../model'):
    # load data
    klgdata = pd.read_csv(datapath)

    ###########################################
    ######## working on latest data ###########
    ###########################################

    df = klgdata[klgdata['flag'] == 1]

    # collect student - knowledge pairs of latest
    stu_col = []
    klg_col = []
    pair_set = collect_pairs(df)
    for pair in pair_set:
        stu_col.append(pair[0])
        klg_col.append(pair[1])

    # build new dataframe
    data = {'stu_user_id': np.array(stu_col), 'knowledge_ids': np.array(klg_col)}
    model = pd.DataFrame(data=data)

    # build remaining column and json data

    json_data = {}

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
    current_student_scores["knowledgeScores"] = knowledgScores
    json_data[stu] = current_student_scores

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
            current_student_scores["knowledgeScores"] = knowledgScores
            json_data[stu] = current_student_scores

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
    model['flag'] = 1

    ###########################################
    ######## working on filled data ###########
    ###########################################

    fill = klgdata[klgdata['flag'] == 0]

    stu_col = []
    klg_col = []
    pair_set = collect_pairs(fill)
    for pair in pair_set:
        stu_col.append(pair[0])
        klg_col.append(pair[1])

    # build new dataframe
    data = {'stu_user_id': np.array(stu_col), 'knowledge_ids': np.array(klg_col)}
    extn = pd.DataFrame(data=data)

    ## initialization
    stu = extn['stu_user_id'].iloc[0]
    summary = get_summary(fill, stu)
    klg = extn['knowledge_ids'].iloc[0]

    ### record info into cols
    avg_score_col = [summary[0][klg]]
    col_0 = [summary[1][klg]]
    col_1 = [summary[2][klg]]
    col_sing = [summary[3][klg]]
    col_comp = [summary[4][klg]]

    ### record info into json data
    knowledgScores = []
    avg_scores = summary[0]
    for (key, val) in avg_scores.items():
        knowledgScores.append({"knowledgeTagId": key, "score": val})
    if stu in json_data.keys():
        json_data[stu]['knowledgeScores'] += knowledgScores
    else:
        current_student_scores = {"stuUserId": str(stu)}
        current_student_scores['knowledgeScores'] = knowledgScores
        json_data[stu] = current_student_scores

    ## loop
    for i in range(1, extn.shape[0]):
        stu = extn['stu_user_id'].iloc[i]

        if stu != extn['stu_user_id'].iloc[i-1]: # need to change summary
            summary = get_summary(fill, stu)
            ### record json data
            knowledgScores = []
            avg_scores = summary[0]
            for (key, val) in avg_scores.items():
                knowledgScores.append({"knowledgeTagId": key, "score": val})
            if stu in json_data.keys():
                json_data[stu]['knowledgeScores'] += knowledgScores
            else:
                current_student_scores = {"stuUserId": str(stu)}
                current_student_scores['knowledgeScores'] = knowledgScores
                json_data[stu] = current_student_scores

        klg = extn['knowledge_ids'].iloc[i]
        avg_score_col.append(summary[0][klg])
        col_0.append(summary[1][klg])
        col_1.append(summary[2][klg])
        col_sing.append(summary[3][klg])
        col_comp.append(summary[4][klg])

    ## build columns
    extn['score_rate'] = np.array(avg_score_col)
    extn['1_score_count'] = np.array(col_1)
    extn['0_score_count'] = np.array(col_0)
    extn['single_knowledge'] = np.array(col_sing)
    extn['multiple_knowledge'] = np.array(col_comp)
    extn['total_count'] = extn['single_knowledge'] + extn['multiple_knowledge']
    extn['flag'] = 0

    ###########################################
    ######## formatting & averaging ###########
    ###########################################
    model = pd.concat([model,extn], ignore_index=True)

    # sort the values by stu id and klg id
    model = model.sort_values(by = ['stu_user_id', 'knowledge_ids'], ascending=(True, True))

    # calculate klg average, write into json
    model = extract_klg_avg(model, folder)

    # calculate per student average, write into json
    model = extract_stu_avg(model, folder)

    # format the knowledge id as json 
    for i in range(model.shape[0]):
        model['knowledge_ids'].iloc[i] = json.dumps([model['knowledge_ids'].iloc[i]])

    # generate student_knowledge file, csv & json
    extract_stu_klg(model)
    extract_stu_klg_json(json_data)

    # set the index as stu id
    model = model.set_index("stu_user_id", drop=True)

    # save the dataframe to file
    model.to_csv(folder + "/model.csv")


def check_folder(folder = '../model'):
    if not os.path.exists('../config'):
        os.mkdir('../config')
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists('../result'):
        os.mkdir('../result')


if __name__ == '__main__':
    # the first component of the string typed
    data_file = sys.argv[1]
    if not data_file.endswith('.csv'):
        print('wrong file type')
        exit(1)

    check_folder()
    create_statistic_model(data_file)

        
