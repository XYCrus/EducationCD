#%%
import sys
import time

import pandas as pd
import numpy as np
import json
import os.path

start = time.time()

student_average_file = "/student_average.json"
knowledge_average_file = "/knowledge_average.json"
model_file = "/model.csv"
# prediction_set_file="/prediction_set.csv"
result_file = "/result.csv"


#%%
def create_df_and_set(model_path, prediction_set_path):
    model_df = pd.read_csv(model_path + model_file).iloc[:, 0:3]
    stu_set = set(list(model_df["stuUserId"]))
    klg_set = set(json.loads(i)[0] for i in list(model_df["knowledgeTagIds"]))
    model_df = model_df.set_index(['stuUserId', 'knowledgeTagIds'])
    stu_klg_pair_set=set(model_df.index)
    pred_df = pd.read_csv(prediction_set_path)
    pred_df_copy = pred_df[["stuUserId", "examinationId", "startDatetime","questionType", "knowledgeTagIds", "scorePercentage"]].copy()


    # record of known/unknown klg and stu
    # known stu 1, else 0
    pred_df_copy["studentFlag"] = None
    # all klg are known:1, else 0
    pred_df_copy["knowledgeFlag"] = None
    # at least one klg known:1, else 0
    pred_df_copy["halfKnowledgeFlag"] = None
    # record of prediction
    pred_df_copy["predictionBeforeMapping"] = None
    pred_df_copy["predictionAfterMapping"] = None
    return model_df, pred_df_copy, stu_set, klg_set,stu_klg_pair_set


# score of a specified student's knowledge
def get_avg_score_of_stu_klg(df: pd.DataFrame, stu_user_id: int, knowledge_id):
    res = df.loc[(stu_user_id, knowledge_id)]
    score = res.values[0][0]
    return score


def check_folder():
    if not os.path.exists('../model'):
        os.mkdir('../model')
    if not os.path.exists('../result'):
        os.mkdir('../result')


def generate_student_knowledge_flag(student_set, knowledge_set, stuUserId, knowledgeTagIds,knowledge_num):
    num_of_known_knowledge = 0
    # check if student is new
    if stuUserId in student_set:
        studentFlag = 1
    elif stuUserId not in student_set:
        studentFlag = 0
    # check if knowledge is new
    for klg in knowledgeTagIds:
        if klg in knowledge_set:
            num_of_known_knowledge += 1
        # pred_df_copy.loc[index, f"klg{j}"] = klg
    if num_of_known_knowledge == knowledge_num:
        knowledgeFlag = 1
        halfKnowledgeFlag = 1
    elif knowledge_num > num_of_known_knowledge > 0:
        knowledgeFlag = 0
        halfKnowledgeFlag = 1
    elif num_of_known_knowledge == 0:
        knowledgeFlag = 0
        halfKnowledgeFlag = 0
    return studentFlag, knowledgeFlag, halfKnowledgeFlag


def predict_student_scores_before_mapping(student_set, knowledge_set,knowledge_student_pair_set, stuUserId, knowledgeTagIds,knowledge_num,student_average_dict,knowledge_average_dict):
    # prediction
    single_stu_score = []
    # check if the student is known
    if stuUserId in student_set:
        # iterate over klg, with maximun number of klg=3
        for j in range(knowledge_num):
            knowledge_id = knowledgeTagIds[j]
            # check if the klg is known
            if knowledge_id in knowledge_set:
                klg_id_js = json.dumps(knowledge_id.split(" "))
                if (stuUserId,klg_id_js) in knowledge_student_pair_set:
                    score = get_avg_score_of_stu_klg(model_df, stu_user_id=stuUserId, knowledge_id=klg_id_js)
                else:

                    score=knowledge_average_dict[int(knowledge_id)]

                if score is not None:
                    single_stu_score.append(score)
                # student miss some of the exam
                else:
                    single_stu_score.append(knowledge_average_dict[int(knowledge_id)])
            # when klg_id is not none and is unknown
            elif knowledge_id not in knowledge_set:
                single_stu_score.append(student_average_dict[stuUserId])
        predictionBeforeMapping = min(single_stu_score)
    elif stuUserId not in student_set:
        for j in range(knowledge_num):
            knowledge_id = knowledgeTagIds[j]
            # check if the klg is known
            if knowledge_id in knowledge_set:
                single_stu_score.append(knowledge_average_dict[int(knowledge_id)])
            # when klg_id is not none and is unknown
            elif knowledge_id not in knowledge_set:
                single_stu_score.append(total_average)
        predictionBeforeMapping = min(single_stu_score)
    return predictionBeforeMapping


def predict_student_scores_after_mapping(question_type, predictionBeforeMapping):
    if question_type == "SCHOICE":
        predictionAfterMapping = 1 if predictionBeforeMapping > 0.5 else 0
    elif question_type == "FILLBLANK" or question_type == "MCHOICE":
        if predictionBeforeMapping >= 0.7:
            predictionAfterMapping = 1
        elif predictionBeforeMapping > 0.5:
            predictionAfterMapping = 0.6
        elif predictionBeforeMapping > 0.3:
            predictionAfterMapping = 0.4
        else:
            predictionAfterMapping = 0
    elif question_type == "SHORTANSWER":
        predictionAfterMapping = predictionBeforeMapping
    return predictionAfterMapping

#%%
if __name__ == '__main__':
    # sys.argv catches the command parameters typed (sep with ' ')
    # read in and record parameters
    check_folder()
    prediction_set_path = sys.argv[1]
    result_path = sys.argv[2]
    model_path = sys.argv[3]

    if not (prediction_set_path.endswith('.csv')):
        print('wrong file type')
        exit(1)
    # get all df needed
    df_res = create_df_and_set(model_path, prediction_set_path)
    model_df = df_res[0]
    prediction_df_copy = df_res[1]
    student_set = df_res[2]
    knowledge_set = df_res[3]
    knowledge_student_pair_set=df_res[4]
    total_average = 0.5
    student_average_dict = json.load(open(model_path + student_average_file, 'r', encoding="utf-8"))
    student_average_dict = {int(key): value for key, value in student_average_dict.items()}
    knowledge_average_dict = json.load(open(model_path + knowledge_average_file, 'r', encoding="utf-8"))
    knowledge_average_dict = {int(key): value for key, value in knowledge_average_dict.items()}

    # loop
    for index, row in prediction_df_copy.iterrows():
        knowledgeTagIds = json.loads(row['knowledgeTagIds'])
        knowledge_num = len(knowledgeTagIds)
        stuUserId = row["stuUserId"]
        # flag known/unknown student and klg
        flag_result = generate_student_knowledge_flag(student_set, knowledge_set, stuUserId, knowledgeTagIds,knowledge_num)
        prediction_df_copy.loc[index, "studentFlag"] = flag_result[0]
        prediction_df_copy.loc[index, "knowledgeFlag"] = flag_result[1]
        prediction_df_copy.loc[index, "halfKnowledgeFlag"] = flag_result[2]

        # prediction
        predictionBeforeMapping=predict_student_scores_before_mapping(student_set,knowledge_set,knowledge_student_pair_set,stuUserId,knowledgeTagIds,knowledge_num,student_average_dict,knowledge_average_dict)
        prediction_df_copy.loc[index, "predictionBeforeMapping"] = predictionBeforeMapping
        # Mapping
        question_type = row["questionType"]
        predictionAfterMapping=predict_student_scores_after_mapping(question_type, predictionBeforeMapping)
        prediction_df_copy.loc[index, "predictionAfterMapping"] = predictionAfterMapping
    # check if prediction is accurate
    prediction_df_copy["accuracyFlag"] = None
    for index, row in prediction_df_copy.iterrows():
        qt = row['questionType']
        answer_score = row['scorePercentage']
        predictionAfterMapping = row['predictionAfterMapping']
        if abs(answer_score - predictionAfterMapping) <= 0.3:
            answer_accuracy = 1
        else:
            answer_accuracy = 0
        prediction_df_copy.loc[index, 'accuracyFlag'] = answer_accuracy

    accu = np.array(prediction_df_copy["accuracyFlag"])
    accu_purified = np.array([i for i in accu if i is not None])
    accuracy = accu_purified.mean()

    prediction_df_copy["differences"] = prediction_df_copy["predictionAfterMapping"] - prediction_df_copy[
        "scorePercentage"]

    result = prediction_df_copy[
        ["stuUserId", "examinationId", "startDatetime","knowledgeTagIds", "questionType", "scorePercentage", "studentFlag",
         "knowledgeFlag",
         "halfKnowledgeFlag", "predictionBeforeMapping", "predictionAfterMapping", "differences", "accuracyFlag"]]

    result.to_csv(result_path + result_file, index=False)

end = time.time()
print('Running time: %s Seconds' % (end - start))
