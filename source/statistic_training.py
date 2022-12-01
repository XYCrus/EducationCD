import string
import sys
import os.path
import pandas as pd
import numpy as np
import json
from datetime import datetime


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


def extract_stu_klg(model, result_folder='../result'):
    # save as csv file
    stu_klg = pd.DataFrame(model, columns=['stuUserId', 'knowledgeTagIds', 'scorePercentage'])
    stu_klg = stu_klg.set_index("stuUserId", drop=True)
    stu_klg.to_csv(result_folder + "/student_knowledge_statistics.csv")


def extract_stu_klg_json(json_data, result_folder='../result'):
    json_data = list(json_data.values())
    with open(result_folder + "/student_knowledge_statistics.json", "w") as outfile:
        json.dump(json_data, outfile, indent=4)


def extract_klg_avg(model: pd.DataFrame, model_folder='../model', result_folder='../result'):
    df = model.copy()
    df = df.set_index('knowledgeTagIds', drop=False)
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

    with open(result_folder + "/knowledge_average.json", "w") as outfile:
        json.dump(klg_avg_dict, outfile, indent=4)

    with open(model_folder + "/knowledge_average.json", "w") as outfile:
        json.dump(klg_avg_dict, outfile, indent=4)

    return model


def extract_stu_avg(model: pd.DataFrame, model_folder='../model', result_folder='../result'):
    df = model.copy()
    df = df.set_index('stuUserId', drop=False)
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

    with open(result_folder + "/student_average.json", "w") as outfile:
        json.dump(stu_score, outfile, indent=4)

    with open(model_folder + "/student_average.json", "w") as outfile:
        json.dump(stu_score, outfile, indent=4)

    return model


def create_statistic_model(datapath: string, model_folder='../model', result_folder='../result'):
    # load data
    klgdata = pd.read_csv(datapath)

    ###########################################
    ######## working on latest data ###########
    ###########################################

    df = klgdata[klgdata['flag'] == 1]

    # find all unique student ids
    all_stu = df['stuUserId'].unique()
    all_stu

    # initialization
    stu = all_stu[0]
    avg_scores, count_0, count_1, count_single, count_multiple = get_summary(df, stu)
    ## initialize model dataframe
    klg_col, score_col = list(avg_scores.keys()), list(avg_scores.values())
    stu_col = [stu] * len(klg_col)
    count_1_col = list(count_1.values())
    count_0_col = list(count_0.values())
    count_single_col = list(count_single.values())
    count_multi_col = list(count_multiple.values())
    count_tot = np.array(count_single_col) + np.array(count_multi_col)

    data = {'stuUserId': stu_col, 'knowledgeTagIds': klg_col, 'scorePercentage': score_col,
            '1_score_count': count_1_col,
            '0_score_count': count_0_col, 'single_knowledge': count_single_col, 'multiple_knowledge': count_multi_col,
            'total_count': count_tot}
    model = pd.DataFrame(data=data)

    ## initialize json data
    json_data = {}
    ## record info into json data
    current_student_scores = {"stuUserId": str(stu)}
    knowledgScores = []
    for (key, val) in avg_scores.items():
        knowledgScores.append({"knowledgeTagId": key, "score": val})
    current_student_scores["knowledgeScores"] = knowledgScores
    json_data[stu] = current_student_scores

    # loop
    for i in range(1, len(all_stu)):
        stu = all_stu[i]
        ## write chunk
        avg_scores, count_0, count_1, count_single, count_multiple = get_summary(df, stu)

        klg_col, score_col = list(avg_scores.keys()), list(avg_scores.values())
        stu_col = [stu] * len(klg_col)
        count_1_col = list(count_1.values())
        count_0_col = list(count_0.values())
        count_single_col = list(count_single.values())
        count_multi_col = list(count_multiple.values())
        count_tot = np.array(count_single_col) + np.array(count_multi_col)

        data = {'stuUserId': stu_col, 'knowledgeTagIds': klg_col, 'scorePercentage': score_col,
                '1_score_count': count_1_col,
                '0_score_count': count_0_col, 'single_knowledge': count_single_col,
                'multiple_knowledge': count_multi_col,
                'total_count': count_tot}
        chunk = pd.DataFrame(data=data)

        ## append chunk to model
        model = pd.concat([model, chunk], ignore_index=True)

        ## record json data
        current_student_scores = {"stuUserId": str(stu)}
        knowledgScores = []
        for (key, val) in avg_scores.items():
            knowledgScores.append({"knowledgeTagId": key, "score": val})
        current_student_scores["knowledgeScores"] = knowledgScores
        json_data[stu] = current_student_scores

        # finishing latest data
        model['flag'] = 1

    ###########################################
    ######## working on filled data ###########
    ###########################################

    fill_df = klgdata[klgdata['flag'] == 0]

    # find all unique student ids
    all_stu = fill_df['stuUserId'].unique()

    # initialization
    stu = all_stu[0]
    avg_scores, count_0, count_1, count_single, count_multiple = get_summary(fill_df, stu)
    ## initialize model dataframe
    klg_col, score_col = list(avg_scores.keys()), list(avg_scores.values())
    stu_col = [stu] * len(klg_col)
    count_1_col = list(count_1.values())
    count_0_col = list(count_0.values())
    count_single_col = list(count_single.values())
    count_multi_col = list(count_multiple.values())
    count_tot = np.array(count_single_col) + np.array(count_multi_col)

    data = {'stuUserId': stu_col, 'knowledgeTagIds': klg_col, 'scorePercentage': score_col,
            '1_score_count': count_1_col,
            '0_score_count': count_0_col, 'single_knowledge': count_single_col, 'multiple_knowledge': count_multi_col,
            'total_count': count_tot}
    model_extension = pd.DataFrame(data=data)

    ## record info into json data
    knowledgScores = []
    for (key, val) in avg_scores.items():
        knowledgScores.append({"knowledgeTagId": key, "score": val})
    if stu in json_data.keys():
        json_data[stu]['knowledgeScores'] += knowledgScores
    else:
        current_student_scores = {"stuUserId": str(stu)}
        current_student_scores['knowledgeScores'] = knowledgScores
        json_data[stu] = current_student_scores

    # loop
    for i in range(1, len(all_stu)):
        stu = all_stu[i]
        ## write chunk
        avg_scores, count_0, count_1, count_single, count_multiple = get_summary(fill_df, stu)

        klg_col, score_col = list(avg_scores.keys()), list(avg_scores.values())
        stu_col = [stu] * len(klg_col)
        count_1_col = list(count_1.values())
        count_0_col = list(count_0.values())
        count_single_col = list(count_single.values())
        count_multi_col = list(count_multiple.values())
        count_tot = np.array(count_single_col) + np.array(count_multi_col)

        data = {'stuUserId': stu_col, 'knowledgeTagIds': klg_col, 'scorePercentage': score_col,
                '1_score_count': count_1_col,
                '0_score_count': count_0_col, 'single_knowledge': count_single_col,
                'multiple_knowledge': count_multi_col,
                'total_count': count_tot}
        chunk = pd.DataFrame(data=data)

        ## append chunk to model_extension
        model_extension = pd.concat([model_extension, chunk], ignore_index=True)

        ## record json data
        knowledgScores = []
        for (key, val) in avg_scores.items():
            knowledgScores.append({"knowledgeTagId": key, "score": val})
        if stu in json_data.keys():
            json_data[stu]['knowledgeScores'] += knowledgScores
        else:
            current_student_scores = {"stuUserId": str(stu)}
            current_student_scores['knowledgeScores'] = knowledgScores
            json_data[stu] = current_student_scores

    # finishing filled data
    model_extension['flag'] = 0

    # integrate model and model_extension
    model = pd.concat([model, model_extension], ignore_index=True)

    ###########################################
    ######## formatting & averaging ###########
    ###########################################

    # sort the values by stu id and klg id
    model = model.sort_values(by=['stuUserId', 'knowledgeTagIds'], ascending=(True, True))

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


def check_folder(model_folder='../model', result_folder='../result'):
    if not os.path.exists('../config'):
        os.mkdir('../config')
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)


if __name__ == '__main__':
    begin = datetime.now()
    # the first component of the string typed
    data_file = sys.argv[1]
    if not data_file.endswith('.csv'):
        print('wrong file type')
        exit(1)

    result_folder = sys.argv[2]
    model_folder = sys.argv[3]

    check_folder(model_folder, result_folder)
    create_statistic_model(data_file, model_folder, result_folder)
    end = datetime.now()
    print("time: ", end - begin)
