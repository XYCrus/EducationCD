import json

#Program for generating each student with every knowledge, and fake question ids

class FakeTestGenerator:

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

        len1 = len(exer_n) - 1
        first = exer_n[0]
        new_start = (int(first) + 1) * (10 ** len1)
        fake_exer_id = new_start

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
        elif original_file.endswith('.csv'):
            for stu in stu_list:
                for kl in knowledge_list:
                    sample = {
                        "stu_user_id": stu,
                        "question_id": fake_exer_id,
                        "score_percentage": -1,
                        "knowledge_ids": [
                            kl
                        ]
                    }
                    fake_train.append(sample)
                    fake_exer_id += 1
                fake_exer_id = new_start
        
        return fake_train    