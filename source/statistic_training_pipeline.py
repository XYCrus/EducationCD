#%%
import sys
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import os.path

#%%
def collect_pairs(datapath, df = True):
    if df:
        df = pd.read_csv(datapath)
    else:
        df = datapath
    pair_set = set([])
    for i in range(df.shape[0]):
        stu = df['stuUserId'].iloc[i]
        knowledges = json.loads(df['knowledgeTagIds'].iloc[i])
        for klg in knowledges:
            pair_set.add((stu,klg))
    return pair_set

def get_summary(df, stu_id):
    sub_df = df[df['stuUserId'] == stu_id]
    scores, count_1, count_0, single_apr, complex_apr = {}, {}, {}, {}, {}
    
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


def extract_stu_klg(model, 
                    result_folder='../result', 
                    write = True):
    stu_klg = pd.DataFrame(model, columns=['stuUserId', 
                                           'knowledgeTagIds', 
                                           'scorePercentage'])
    
    stu_klg = stu_klg.set_index("stuUserId", drop=True)
    if write:
        stu_klg.to_csv(result_folder + "/student_knowledge_statistics.csv")


def extract_stu_klg_json(json_data, 
                         result_folder='../result', 
                         write = True):
    json_data = list(json_data.values())
    if write:
        with open(result_folder + "/student_knowledge_statistics.json", 
                  "w") as outfile:
            json.dump(json_data, outfile, indent=4)


def extract_klg_avg(model: pd.DataFrame, 
                    model_folder='../model', 
                    result_folder='../result', 
                    write = True):
    df = model.copy()
    df = df.set_index('knowledgeTagIds', drop=False)
    grouped = df['scorePercentage'].groupby('knowledgeTagIds').mean()
    model['klg_avg'] = model['knowledgeTagIds'].apply(lambda x: grouped.loc[x])
    klg_avg_dict = dict(grouped)

    if write:
        with open(result_folder + "/knowledge_average.json", "w") as outfile:
            json.dump(klg_avg_dict, outfile, indent=4)
        with open(model_folder + "/knowledge_average.json", "w") as outfile:
            json.dump(klg_avg_dict, outfile, indent=4)
    return model


def extract_stu_avg(model: pd.DataFrame, 
                    model_folder='../model', 
                    result_folder='../result', 
                    write = True):
    df = model.copy()
    df = df.set_index('stuUserId', drop=False)
    grouped = df['scorePercentage'].groupby('stuUserId').mean()
    model['stu_avg'] = model['stuUserId'].apply(lambda x: grouped.loc[x])
    stu_score = dict(grouped)
    if write:
        with open(result_folder + "/student_average.json", "w") as outfile:
            json.dump(stu_score, outfile, indent=4)
        with open(model_folder + "/student_average.json", "w") as outfile:
            json.dump(stu_score, outfile, indent=4)
        return model
    else:
        stringoutput = json.dumps(stu_score, indent=4)
        return stringoutput
    
def extract_common(model, model_folder, result_folder, boo):
    model = extract_klg_avg(model, model_folder, result_folder, boo)
    model = extract_stu_avg(model, model_folder, result_folder, boo)
    
    return model
    
def extract_extra(model, js_data, folderm, folderr):
    for i in range(model.shape[0]):
        model['knowledgeTagIds'].iloc[i] = json.dumps(
            [model['knowledgeTagIds'].iloc[i]])
        
    extract_stu_klg(model, folderr, True)
    extract_stu_klg_json(js_data, folderr, True)

    model = model.set_index("stuUserId", drop=True)

    model.to_csv(folderm + "/model.csv")
    
    return model
    
def read(filename, OutputFileType = 'json'):
    wholedata = pd.read_csv(filename)
    
    if not os.path.exists('../result'):
        os.mkdir('../result')
        
    if OutputFileType == 'json':
        result = wholedata.to_json(orient="columns")
        parsed = json.loads(result)
        #json.dumps(parsed, indent=4)
    
        with open("../result/format.json", "w") as outfile:
            json.dump(parsed, outfile, indent=4)
        return result
    
    elif OutputFileType == 'csv':
        if not os.path.exists('../model'):
            os.mkdir('../model')
            
        return wholedata

def json_to_df(jsFile):
    jsonstring = json.loads(jsFile)
    wholedata = pd.DataFrame.from_dict(jsonstring)
    return wholedata
    
def run(Input, 
        Output = False,
        model_folder = '../model', 
        result_folder = '../result', 
        n_latest = 3, 
        n_fill = 3):
    
    if not Output:
        wholedata = json_to_df(Input)
    else:
        wholedata = Input

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

    data = {'stuUserId': stu_col, 
            'knowledgeTagIds': klg_col, 
            'scorePercentage': score_col,
            '1_score_count': count_1_col,
            '0_score_count': count_0_col, 
            'single_knowledge': count_single_col, 
            'multiple_knowledge': count_multi_col,
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

        data = {'stuUserId': stu_col, 
                'knowledgeTagIds': klg_col, 
                'scorePercentage': score_col,
                '1_score_count': count_1_col,
                '0_score_count': count_0_col, 
                'single_knowledge': count_single_col,
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

    data = {'stuUserId': stu_col, 
            'knowledgeTagIds': klg_col, 
            'scorePercentage': score_col,
            '1_score_count': count_1_col,
            '0_score_count': count_0_col, 
            'single_knowledge': count_single_col, 
            'multiple_knowledge': count_multi_col,
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

        data = {'stuUserId': stu_col, 
                'knowledgeTagIds': klg_col, 
                'scorePercentage': score_col,
                '1_score_count': count_1_col,
                '0_score_count': count_0_col, 
                'single_knowledge': count_single_col,
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
    
    
    if not Output:
    
        return extract_common(model, model_folder, result_folder, False)
    
    else:
        model = extract_common(model, model_folder, result_folder, True)
        
        return extract_extra(model, json_data, model_folder, result_folder)
    
#%%
if __name__ == '__main__':
    begin = datetime.now()

    if len(sys.argv) == 4: 
        filename = sys.argv[1]
        stringinput = read(filename, OutputFileType = 'csv')
        stringoutput = run(stringinput, Output = True)
        
    # no output
    elif len(sys.argv) == 2:  
        filename = sys.argv[1]
        stringinput = read(filename, OutputFileType = 'json')
        stringoutput = run(stringinput, Output = False)
        #print(stringoutput)
        #print(type(stringoutput))
        
    end = datetime.now()
    print("time: ", end - begin)
    