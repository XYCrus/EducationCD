# SY Hong
# Time:10/15/2022 2:08 PM

import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error

previous_month_df = pd.read_csv("data/latest_234.csv")
model_df = pd.read_csv("data/model(1).csv").iloc[:, 0:4]
pred_df = pd.read_csv("data/prediction_set.csv")
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

stu_set = set(list(model_df["stu_user_id"]))
klg_set = set(list(model_df["knowledge_id"]))


# score of a specified student's knowledge
def get_avg_score_of_stu_klg(df: pd.DataFrame, stu_user_id: int, knowledge_id: int):
    for i in range(len(df)):
        if df["stu_user_id"].iloc[i] == stu_user_id and df["knowledge_id"].iloc[i] == knowledge_id:
            return df["score_rate"].iloc[i]


# average score of a student in previous month
def get_avg_score_of_stu(df: pd.DataFrame):
    mean = df["scorePercentage"].groupby(df["stuUserId"]).mean()
    stu_avg_dict = dict(mean)
    return stu_avg_dict


# average score of knowledge in previous month
def get_avg_score_of_klg(df: pd.DataFrame):
    mean = df["score_rate"].groupby(df["knowledge_id"]).mean()
    klg_avg_dict = dict(mean)
    return klg_avg_dict


# average score of all students
def get_total_avg_score(df: pd.DataFrame):
    return np.mean(df["scorePercentage"])



total_avg = get_total_avg_score(previous_month_df)
stu_avg_dict = get_avg_score_of_stu(previous_month_df)
klg_avg_dict = get_avg_score_of_klg(model_df)

# split knowledge
for i in range(pred_df_copy.shape[0]):
    knowledges = json.loads(pred_df['knowledgeTagIds'].iloc[i])
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
            if 1.2 * (pred_df_copy['scorePercentage'].iloc[i]) >= pred_df_copy['pred_after_mapping'].iloc[i] >= 0.8 * (
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

res.to_csv("res_new.csv", index=False)

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
