
import sys
import pandas as pd
import numpy as np
import json
import os.path


stu_avg_file = "/student_average.json"
klg_avg_file= "/knowledge_average.json"
model_file = "/model.csv"
#prediction_set_file="/prediction_set.csv"
result_file="/result.csv"
def create_df_and_set(model_path, prediction_sSet_path):
    model_df = pd.read_csv(model_path + model_file).iloc[:, 0:3]
    stu_set = set(list(model_df["stuUserId"]))
    klg_set = set(list(model_df["knowledgeTagIds"]))
    model_df = model_df.set_index(['stuUserId', 'knowledgeTagIds'])
    pred_df = pd.read_csv(prediction_set_path)
    pred_df_copy = pred_df[["stuUserId", "examinationId", "questionType", "knowledgeTagIds", "scorePercentage"]].copy()
    # record splited klg
    pred_df_copy[f"klg{0}"] = None
    pred_df_copy[f"klg{1}"] = None
    pred_df_copy[f"klg{2}"] = None

    # record of known/unknown klg and stu
    # known stu 1, else 0
    pred_df_copy["stu_flag"] = None
    # all klg are known:1, else 0
    pred_df_copy["klg_flag"] = None
    # at least one klg known:1, else 0
    pred_df_copy["half_klg_flag"] = None
    # record of prediction
    pred_df_copy["pred_before_mapping"] = None
    pred_df_copy["pred_after_mapping"] = None
    return model_df, pred_df_copy, stu_set, klg_set


# score of a specified student's knowledge
def get_avg_score_of_stu_klg(df: pd.DataFrame, stu_user_id: int, knowledge_id):
    res = df.loc[(stu_user_id, knowledge_id)]
    score=res.values[0][0]
    return score

# average score of all students
def get_total_avg_score(df: pd.DataFrame):
    return np.mean(df["score_rate"])


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
    check_folder()
    # the first component of the string typed
    model_path = sys.argv[1]
    prediction_set_path = sys.argv[2]
    result_path=sys.argv[3]
    # if not (prediction_set_path.endswith('.csv')):
    #     print('wrong file type')
    #     exit(1)
    # get all df needed
    df_res = create_df_and_set(model_path, prediction_set_path)
    model_df = df_res[0]
    pred_df_copy = df_res[1]
    stu_set = df_res[2]
    klg_set = df_res[3]
    # total_avg = get_total_avg_score(model_df)
    total_avg=0.5
    stu_avg_dict = json.load(open(model_path + stu_avg_file, 'r', encoding="utf-8"))
    stu_avg_dict = {int(key): value for key, value in stu_avg_dict.items()}
    klg_avg_dict = json.load(open(model_path + klg_avg_file, 'r', encoding="utf-8"))
    klg_avg_dict = {int(key): value for key, value in klg_avg_dict.items()}
    # split knowledge
    for i in range(pred_df_copy.shape[0]):
        knowledges = json.loads(pred_df_copy['knowledgeTagIds'].iloc[i])
        for j in range(len(knowledges)):
            pred_df_copy.loc[i, f"klg{j}"] = json.dumps([knowledges[j]])

    # flag of known/unknown student and klg
    for i in range(pred_df_copy.shape[0]):
        stu_user_id = pred_df_copy["stuUserId"].iloc[i]
        klg_list = []
        num_of_known_klg = 0
        if stu_user_id in stu_set:
            pred_df_copy.loc[i, "stu_flag"] = 1
        elif stu_user_id not in stu_set:
            pred_df_copy.loc[i, "stu_flag"] = 0
        for j in range(3):
            if pred_df_copy[f"klg{j}"].iloc[i] is not None:
                klg_list.append(pred_df_copy[f"klg{j}"].iloc[i])
        for k in range(len(klg_list)):
            if klg_list[k] in klg_set:
                num_of_known_klg += 1
        if num_of_known_klg == len(klg_list):
            pred_df_copy.loc[i, "klg_flag"] = 1
            pred_df_copy.loc[i, "half_klg_flag"] = 1
        elif num_of_known_klg < len(klg_list) and num_of_known_klg > 0:
            pred_df_copy.loc[i, "klg_flag"] = 0
            pred_df_copy.loc[i, "half_klg_flag"] = 1
        elif num_of_known_klg == 0:
            pred_df_copy.loc[i, "klg_flag"] = 0
            pred_df_copy.loc[i, "half_klg_flag"] = 0
    # prediction
    for i in range(pred_df_copy.shape[0]):
        stu_user_id = pred_df_copy["stuUserId"].iloc[i]
        single_stu_score = []
        # check if the student is known
        if stu_user_id in stu_set:
            # iterate over klg, with maximun number of klg=3
            for j in range(3):
                klg_id = pred_df_copy[f"klg{j}"].iloc[i]
                # check if the klg is known
                if klg_id in klg_set:
                    if get_avg_score_of_stu_klg(model_df, stu_user_id=stu_user_id, knowledge_id=klg_id) is not None:
                        single_stu_score.append(
                            get_avg_score_of_stu_klg(model_df, stu_user_id=stu_user_id, knowledge_id=klg_id))
                    # student miss some of the exam
                    else:
                        single_stu_score.append(klg_avg_dict[klg_id])
                # when klg_id is not none and is unknown
                elif klg_id is not None and klg_id not in klg_set:
                    single_stu_score.append(stu_avg_dict[stu_user_id])

            pred_df_copy.loc[i, "pred_before_mapping"] = min(single_stu_score)
        elif stu_user_id not in stu_set:
            for j in range(3):
                klg_id = pred_df_copy[f"klg{j}"].iloc[i]
                # check if the klg is known
                if klg_id in klg_set:
                    single_stu_score.append(klg_avg_dict[klg_id])
                # when klg_id is not none and is unknown
                elif klg_id is not None and klg_id not in klg_set:
                    single_stu_score.append(total_avg)
            pred_df_copy.loc[i, "pred_before_mapping"] = min(single_stu_score)

    # Mapping
    for i in range(pred_df_copy.shape[0]):
        qt = pred_df_copy["questionType"].iloc[i]
        pred_before_mapping = pred_df_copy["pred_before_mapping"].iloc[i]
        if qt == "SCHOICE":
            pred_df_copy.loc[i, "pred_after_mapping"] = 1 if pred_before_mapping > 0.5 else 0
        elif qt == "FILLBLANK" or qt == "MCHOICE":
            if pred_before_mapping > 0.8:
                pred_df_copy.loc[i, "pred_after_mapping"] = 1
            elif pred_before_mapping > 0.5:
                pred_df_copy.loc[i, "pred_after_mapping"] = 0.6
            elif pred_before_mapping > 0.2:
                pred_df_copy.loc[i, "pred_after_mapping"] = 0.4
            else:
                pred_df_copy.loc[i, "pred_after_mapping"] = 0
        elif qt == "SHORTANSWER":
            pred_df_copy.loc[i, "pred_after_mapping"] = pred_before_mapping

    pred_df_copy["accuracy_flag"] = None
    # check if prediction is accurate
    for i in range(len(pred_df_copy)):
        if pred_df_copy['pred_after_mapping'].iloc[i] is not None:
            if pred_df_copy['questionType'].iloc[i] != "SHORTANSWER":
                if pred_df_copy['scorePercentage'].iloc[i] == pred_df_copy['pred_after_mapping'].iloc[i]:
                    pred_df_copy.loc[i, 'accuracy_flag'] = 1
                else:
                    pred_df_copy.loc[i, 'accuracy_flag'] = 0
            elif pred_df_copy['questionType'].iloc[i] == "SHORTANSWER":
                if 1.2 * (pred_df_copy['scorePercentage'].iloc[i]) >= pred_df_copy['pred_after_mapping'].iloc[
                    i] >= 0.8 * (
                        pred_df_copy['scorePercentage'].iloc[i]):
                    pred_df_copy.loc[i, 'accuracy_flag'] = 1
                else:
                    pred_df_copy.loc[i, 'accuracy_flag'] = 0
    accu = np.array(pred_df_copy["accuracy_flag"])

    accu_purified = np.array([i for i in accu if i is not None])
    accuracy = accu_purified.mean()

    pred_df_copy["differences"] = pred_df_copy["pred_after_mapping"] - pred_df_copy["scorePercentage"]

    res = pred_df_copy[
        ["stuUserId", "examinationId", "knowledgeTagIds", "questionType", "scorePercentage", "stu_flag", "klg_flag",
         "half_klg_flag", "pred_before_mapping", "pred_after_mapping", "differences", "accuracy_flag"]]

    diff = np.array(res["differences"])
    diff_purified = np.array([i for i in diff if not np.isnan(i)])
    sum_square = sum(diff_purified ** 2)
    mse = sum_square / len(res)

    res.to_csv(result_path+result_file, index=False)

    # accuracy of all question type
    all_type_sum = pred_df_copy["accuracy_flag"].groupby(pred_df_copy["questionType"]).sum()
    all_type_count = pred_df_copy["accuracy_flag"].groupby(pred_df_copy["questionType"]).count()
    all_type_sum = dict(all_type_sum)
    all_type_count = dict(all_type_count)

    all_type_accuracy = {}
    for key1, value1 in all_type_sum.items():
        for key2, value2 in all_type_count.items():
            if key1 == key2:
                all_type_accuracy[key1] = value1 / value2

    # mse of all type
    pred_df_copy["squared_differences"] = pred_df_copy["differences"] ** 2

    all_type_squared_diff = pred_df_copy["squared_differences"].groupby(pred_df_copy["questionType"]).sum()
    all_type_squared_diff = dict(all_type_squared_diff)

    all_type_mse = {}
    for key1, value1 in all_type_squared_diff.items():
        for key2, value2 in all_type_count.items():
            if key1 == key2:
                all_type_mse[key1] = value1 / value2

print(accuracy)
print(mse)
print(all_type_accuracy)
print(all_type_mse)
