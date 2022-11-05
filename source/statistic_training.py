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


def extract_stu_klg(model, result_folder = '../result'):
    # save as csv file
    stu_klg = pd.DataFrame(model, columns=['stuUserId','knowledgeTagIds', 'scorePercentage'])
    stu_klg = stu_klg.set_index("stuUserId", drop=True)
    stu_klg.to_csv(result_folder + "/student_knowledge_statistics.csv")


def extract_stu_klg_json(json_data, result_folder = '../result'):
    json_data = list(json_data.values())
    with open(result_folder+"/student_knowledge_statistics.json", "w") as outfile:
        json.dump(json_data, outfile, indent=4)


def extract_klg_avg(model: pd.DataFrame, model_folder = '../model', result_folder = '../result'):
    df = model.copy()
    df = df.set_index('knowledgeTagIds',drop=False)
    grouped = df['scorePercentage'].groupby('knowledgeTagIds').mean()
    model['klg_avg'] = model['knowledgeTagIds'].apply(lambda x: grouped.loc[x])
    klg_avg_dict = dict(grouped)
    
    # klg1_score = {}
    # klg1_count = {}
    # klg0_score = {}
    # klg0_count = {}

    # df1 = model[model['flag'] == 1]
    # for i in range(df1.shape[0]):
    #     klg = df1['knowledgeTagIds'].iloc[i]
    #     score = df1['scorePercentage'].iloc[i]
    #     count = df1['total_count'].iloc[i]
    #     klg1_score[klg] = klg1_score.get(klg,0) + score*count
    #     klg1_count[klg] = klg1_score.get(klg,0) + count
    # for key in klg1_count.keys():
    #     klg1_score[key] = klg1_score[key] / klg1_count[key]

    # df0 = model[model['flag'] == 0]
    # for i in range(df0.shape[0]):
    #     klg = df0['knowledgeTagIds'].iloc[i]
    #     if klg not in klg1_count.keys():
    #         score = df0['scorePercentage'].iloc[i]
    #         count = df0['total_count'].iloc[i]
    #         klg0_score[klg] = klg0_score.get(klg,0) + score*count
    #         klg0_count[klg] = klg0_score.get(klg,0) + count
    # for key in klg0_count.keys():
    #     klg0_score[key] = klg0_score[key] / klg0_count[key]
    
    # klg_avg_dict = dict(klg1_score, **klg0_score)
    # model['klg_avg'] = model['knowledgeTagIds'].apply(lambda x: klg_avg_dict[x])

    with open(result_folder+"/knowledge_average.json", "w") as outfile:
        json.dump(klg_avg_dict, outfile, indent=4)
    

    with open(model_folder+"/knowledge_average.json", "w") as outfile:
        json.dump(klg_avg_dict, outfile, indent=4)

    return model


def extract_stu_avg(model: pd.DataFrame, model_folder = '../model', result_folder = '../result'):
    df = model.copy()
    df = df.set_index('stuUserId',drop=False)
    grouped = df['scorePercentage'].groupby('stuUserId').mean()
    model['stu_avg'] = model['stuUserId'].apply(lambda x: grouped.loc[x])
    stu_score = dict(grouped)
    
    # stu_score = {}
    # stu_count = {}
    # stu_flag = {}

    # df1 = model[model['flag'] == 1]
    # for i in range(df1.shape[0]):
    #     stu = int(df1['stuUserId'].iloc[i])
    #     score = df1['scorePercentage'].iloc[i]
    #     count = df1['total_count'].iloc[i]
    #     stu_score[stu] = stu_score.get(stu,0) + score*count
    #     stu_count[stu] = stu_score.get(stu,0) + count
    #     stu_flag[stu] = 1

    # df0 = model[model['flag'] == 0]
    # for i in range(df0.shape[0]):
    #     stu = int(df0['stuUserId'].iloc[i])
    #     if not stu_flag.get(stu,0):
    #         score = df0['scorePercentage'].iloc[i]
    #         count = df0['total_count'].iloc[i]
    #         stu_score[stu] = stu_score.get(stu,0) + score*count
    #         stu_count[stu] = stu_score.get(stu,0) + count
    # for key in stu_count.keys():
    #     stu_score[key] = stu_score[key] / stu_count[key]
    
    # model['stu_avg'] = model['stuUserId'].apply(lambda x: stu_score[x])
    
    with open(result_folder+"/student_average.json", "w") as outfile:
        json.dump(stu_score, outfile, indent=4)
        
    with open(model_folder+"/student_average.json", "w") as outfile:
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


def create_statistic_model(datapath: string, model_folder = '../model', result_folder = '../result'):
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
    data = {'stuUserId': np.array(stu_col), 'knowledgeTagIds': np.array(klg_col)}
    model = pd.DataFrame(data=data)

    # build remaining column and json data

    json_data = {}

    ## initialization
    stu = model['stuUserId'].iloc[0]
    summary = get_summary(df, stu)
    klg = model['knowledgeTagIds'].iloc[0]

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
        stu = model['stuUserId'].iloc[i]

        if stu != model['stuUserId'].iloc[i-1]: # need to change summary
            summary = get_summary(df, stu)
            ### record json data
            current_student_scores = {"stuUserId": str(stu)}
            knowledgScores = []
            avg_scores = summary[0]
            for (key, val) in avg_scores.items():
                knowledgScores.append({"knowledgeTagId": key, "score": val})
            current_student_scores["knowledgeScores"] = knowledgScores
            json_data[stu] = current_student_scores

        klg = model['knowledgeTagIds'].iloc[i]
        avg_score_col.append(summary[0][klg])
        col_0.append(summary[1][klg])
        col_1.append(summary[2][klg])
        col_sing.append(summary[3][klg])
        col_comp.append(summary[4][klg])

    ## build columns
    model['scorePercentage'] = np.array(avg_score_col)
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
    data = {'stuUserId': np.array(stu_col), 'knowledgeTagIds': np.array(klg_col)}
    extn = pd.DataFrame(data=data)

    ## initialization
    stu = extn['stuUserId'].iloc[0]
    summary = get_summary(fill, stu)
    klg = extn['knowledgeTagIds'].iloc[0]

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
        stu = extn['stuUserId'].iloc[i]

        if stu != extn['stuUserId'].iloc[i-1]: # need to change summary
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

        klg = extn['knowledgeTagIds'].iloc[i]
        avg_score_col.append(summary[0][klg])
        col_0.append(summary[1][klg])
        col_1.append(summary[2][klg])
        col_sing.append(summary[3][klg])
        col_comp.append(summary[4][klg])

    ## build columns
    extn['scorePercentage'] = np.array(avg_score_col)
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
    model = model.sort_values(by = ['stuUserId', 'knowledgeTagIds'], ascending=(True, True))

    # calculate klg average, write into json
    model = extract_klg_avg(model, model_folder, result_folder)

    # calculate per student average, write into json
    model = extract_stu_avg(model, model_folder, result_folder)

    # format the knowledge id as json 
    for i in range(model.shape[0]):
        model['knowledgeTagIds'].iloc[i] = json.dumps([model['knowledgeTagIds'].iloc[i]])

    # generate student_knowledge file, csv & json
    extract_stu_klg(model, result_folder)
    extract_stu_klg_json(json_data, result_folder)

    # set the index as stu id
    model = model.set_index("stuUserId", drop=True)

    # save the dataframe to file
    model.to_csv(model_folder + "/model.csv")


def check_folder(model_folder = '../model', result_folder = '../result'):
    if not os.path.exists('../config'):
        os.mkdir('../config')
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)


if __name__ == '__main__':
    # the first component of the string typed
    data_file = sys.argv[1]
    if not data_file.endswith('.csv'):
        print('wrong file type')
        exit(1)
    
    result_folder = sys.argv[2]
    model_folder = sys.argv[3]

    check_folder(model_folder,result_folder)
    create_statistic_model(data_file, model_folder, result_folder)

        
