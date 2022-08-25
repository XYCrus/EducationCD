import json
import csv
from statistics import mean

# run this program to get the average score of each student

class GetAverage:
    @staticmethod
    def getAverage():

        with open('../result/student_knowledge.csv', "r", encoding="utf-8-sig") as f:
            csv_reader = csv.DictReader(f, skipinitialspace=True)
            stu_data = list(csv_reader)


        with open('../config/stu_map.json', 'r') as inputFile:
            stu_scores = json.load(inputFile)

        for key in stu_scores.keys():
            stu_scores[key] = []

        for log in stu_data:
            stu_scores[log['stuUserId']].append(float(log['scorePercentage']))

        for key in stu_scores.keys():
            stu_scores[key] = mean(stu_scores[key])
        # sort the students base on average score, starting from higher score
        stu_scores = dict(sorted(stu_scores.items(), key=lambda item: item[1], reverse=True))
        output = []
        for key in stu_scores.keys():
            output.append({
                'student': key,
                'average_score': stu_scores[key]
            })

        keys = output[0].keys()
        with open('../result/stu_avg_score.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(output)




