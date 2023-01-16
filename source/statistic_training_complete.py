#%%
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from build_knowledge_dataset import build_dataset,collect_pairs
from statistic_training import create_statistic_model, check_folder, extract_klg_avg, extract_stu_avg,get_summary
from continuous_training import retrain

#%%
def read(filename):
    wholedata = pd.read_csv(filename)
    result = wholedata.to_json(orient="columns")
    return result


def run(wholedata_path, model_folder = '../model', result_folder = '../result', n_latest = 3, n_fill = 3):
    jsonfile = json.loads(wholedata_path)
    wholedata = pd.DataFrame.from_dict(jsonfile)

    exam_dates = np.sort(wholedata['startDatetime'].unique())
    latest_dates = exam_dates[-n_latest:]

    latest = wholedata[wholedata['startDatetime'].isin(latest_dates)]

    data = latest
    
    data['flag'] = 1

    pairs = collect_pairs(wholedata, False) - collect_pairs(latest, False)

    wholedata = wholedata.sort_values(by = 'startDatetime', ascending=False)

    single = {pair:[] for pair in pairs}
    complex = {pair:[] for pair in pairs}

    N = wholedata.shape[0]
    for i in range(N):
        stu = wholedata['stuUserId'].iloc[i]
        klgs = json.loads(wholedata['knowledgeTagIds'].iloc[i])
        for klg in klgs:
            pair = (stu,klg)
            if pair in pairs:
                if len(klgs) == 1:
                    single[pair].append(wholedata.index[i])
                    if len(single[pair]) == n_fill: 
                        pairs.discard(pair)
                else:
                    complex[pair].append(wholedata.index[i])
        if len(pairs) == 0: 
            break
    
    records = {}
    pairs = single.keys()
    for pair in pairs:
        record = single[pair]
        while len(record) < n_fill:
            if complex[pair]:
                record.append(complex[pair].pop(0))
            else:
                break
        records[pair] = record
    
    indices = set([])
    for record in records.values():
        for index in record:
            indices.add(index)
    indices = list(indices)
    
    newdata = wholedata.loc[indices]
    newdata['flag'] = 0
    data = data.append(newdata)

    data = data.sort_values(by = 'stuUserId', ascending=True)

    klgdata = data

    df = klgdata[klgdata['flag'] == 1]

    all_stu = df['stuUserId'].unique()

    stu = all_stu[0]
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

    for i in range(1, len(all_stu)):
        stu = all_stu[i]
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

        model = pd.concat([model, chunk], ignore_index=True)

        current_student_scores = {"stuUserId": str(stu)}
        knowledgScores = []
        for (key, val) in avg_scores.items():
            knowledgScores.append({"knowledgeTagId": key, "score": val})
        current_student_scores["knowledgeScores"] = knowledgScores
        json_data[stu] = current_student_scores

        model['flag'] = 1

    fill_df = klgdata[klgdata['flag'] == 0]

    all_stu = fill_df['stuUserId'].unique()

    stu = all_stu[0]
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
            '0_score_count': count_0_col, 'single_knowledge': count_single_col, 'multiple_knowledge': count_multi_col,
            'total_count': count_tot}
    model_extension = pd.DataFrame(data=data)

    knowledgScores = []
    for (key, val) in avg_scores.items():
        knowledgScores.append({"knowledgeTagId": key, "score": val})
    if stu in json_data.keys():
        json_data[stu]['knowledgeScores'] += knowledgScores
    else:
        current_student_scores = {"stuUserId": str(stu)}
        current_student_scores['knowledgeScores'] = knowledgScores
        json_data[stu] = current_student_scores

    for i in range(1, len(all_stu)):
        stu = all_stu[i]
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

        model_extension = pd.concat([model_extension, chunk], ignore_index=True)

        knowledgScores = []
        for (key, val) in avg_scores.items():
            knowledgScores.append({"knowledgeTagId": key, "score": val})
        if stu in json_data.keys():
            json_data[stu]['knowledgeScores'] += knowledgScores
        else:
            current_student_scores = {"stuUserId": str(stu)}
            current_student_scores['knowledgeScores'] = knowledgScores
            json_data[stu] = current_student_scores

    model_extension['flag'] = 0

    model = pd.concat([model, model_extension], ignore_index=True)

    model = model.sort_values(by=['stuUserId', 'knowledgeTagIds'], ascending=(True, True))

    model = extract_klg_avg(model, model_folder, result_folder, False)

    model = extract_stu_avg(model, model_folder, result_folder, False)

    return model
    
'''
def dummy(filename):
    wholedata = pd.read_csv(filename)
    result = wholedata.to_json(orient="columns")
    jsonfile = json.loads(result)
    wholedata = pd.DataFrame.from_dict(jsonfile)
    return wholedata
'''

#%%
if __name__ == '__main__':
    begin = datetime.now()

    if len(sys.argv) == 4:  
        wholedata_file = sys.argv[1]
        if not (wholedata_file.endswith('.csv')):
            print('wrong file type')
            exit(1)

        result_folder = sys.argv[2]
        model_folder = sys.argv[3]

        check_folder(result_folder=result_folder, model_folder=model_folder)
        build_dataset(wholedata_file, model_folder)
        data_file = model_folder + "/knowledge_dataset.csv"
        create_statistic_model(data_file, model_folder=model_folder, result_folder=result_folder)

    # no output
    elif len(sys.argv) == 2:  
        filename = sys.argv[1]
        stringinput = read(filename)
        stringoutput = run(stringinput)
        #print(stringoutput)
        #print(type(stringoutput))
        
    end = datetime.now()
    print("time: ", end - begin)
