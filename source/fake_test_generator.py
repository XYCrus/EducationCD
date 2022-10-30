import json
import csv

# Program for generating each student with every knowledge, and fake question ids


class FakeTestGenerator:
    @staticmethod
    def generate(original_file):
        
        with open('../config/stu_map.json', encoding='utf8') as i_f:
            stu_map = json.load(i_f)
        with open('../config/knowledge_map.json', encoding='utf8') as i_f:
            knowledge_map = json.load(i_f)

        stu_list = list(stu_map.keys())
        knowledge_list = list(knowledge_map.keys())

        fake_train = []

        with open('../config/config.txt') as configFile:
            student_n, exer_n, knowledge_n = configFile.readline().split(', ')

        # 制造一个new start作为虚拟习题编号
        # 实际习题和虚拟习题编号之间有一段空余
        len1 = len(exer_n) - 1 # the last string index of exer_n
        first = exer_n[0] # the leading nunber of exer_n
        new_start = (int(first) + 1) * (10 ** len1) # e.g. 365 -> 400; 3655 -> 4000
        fake_exer_id = new_start

        # json file处理
        if original_file.endswith('.json'):
            for stu in stu_list:
                for kl in knowledge_list:
                    sample = {
                        "user_id": stu,
                        "exer_id": fake_exer_id,
                        "score": -1,
                        "knowledge_code": [
                            kl
                        ]
                    }
                    fake_train.append(sample)
                    fake_exer_id += 1
                fake_exer_id = new_start
        # csv file处理
        elif original_file.endswith('.csv'):
            # fake a sample for each student and each knowledge
            # for each knowledge, there is a fake excercise entailing it
            # each student 'completes' all of the fake exercises 
            ## and has score = -1 recorded in fake_train
            for stu in stu_list:
                for kl in knowledge_list:
                    sample = {
                        "stuUserId": stu,
                        "questionId": fake_exer_id,
                        "scorePercentage": -1,
                        "knowledgeTagIds": [
                            kl
                        ]
                    }
                    fake_train.append(sample)
                    fake_exer_id += 1

                # insert a fake exercise that contains unseen knowledge (for each student)
                # sample = {
                #     "stuUserId": stu,
                #     "questiond": fake_exer_id,
                #     "scorePercentage": -1,
                #     "knowledgeTagIds": [
                #         '-1'
                #     ]
                # }  
                # fake_train.append(sample)  
                fake_exer_id = new_start
        
        return fake_train
    

    def generate2(original_file):
        
        with open('../config/stu_map.json', encoding='utf8') as i_f:
            stu_map = json.load(i_f)
        with open('../config/knowledge_map.json', encoding='utf8') as i_f:
            knowledge_map = json.load(i_f)
        with open('../config/exer_map.json', encoding='utf8') as i_f:
            exer_map = json.load(i_f)

        stu_list = list(stu_map.keys())
        knowledge_list = list(knowledge_map.keys())

        fake_train = []

        with open('../config/config.txt') as configFile:
            student_n, exer_n, knowledge_n = configFile.readline().split(', ')

        # 制造一个new start作为虚拟习题编号
        # 实际习题和虚拟习题编号之间有一段空余
        len1 = len(exer_n) - 1 # the last string index of exer_n
        first = exer_n[0] # the leading nunber of exer_n
        new_start = (int(first) + 1) * (10 ** len1) # e.g. 365 -> 400; 3655 -> 4000
        fake_exer_id = new_start

        # json file处理
        if original_file.endswith('.json'):
            for stu in stu_list:
                for kl in knowledge_list:
                    sample = {
                        "user_id": stu,
                        "exer_id": fake_exer_id,
                        "score": -1,
                        "knowledge_code": [
                            kl
                        ]
                    }
                    fake_train.append(sample)
                    fake_exer_id += 1
                fake_exer_id = new_start
        # csv file处理
        elif original_file.endswith('.csv'):
            # fake a sample for each student and each knowledge
            # for each knowledge, there is a fake excercise entailing it
            # each student 'completes' all of the fake exercises 
            ## and has score = -1 recorded in fake_train
            with open(original_file, "r", encoding="utf-8-sig") as f: # data_file: whole
                csv_reader = csv.DictReader(f, skipinitialspace=True)
                data1 = list(csv_reader)

            for sample in data1: # each sample is a piece of data
            
                sample["scorePercentage_pre"] = -1
                fake_train.append(sample)

                # insert a fake exercise that contains unseen knowledge (for each student)
                # sample = {
                #     "stuUserId": stu,
                #     "questiond": fake_exer_id,
                #     "scorePercentage": -1,
                #     "knowledgeTagIds": [
                #         '-1'
                #     ]
                # }  
                # fake_train.append(sample)  
        
        return fake_train
