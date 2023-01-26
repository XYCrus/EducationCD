import unittest
from index import handler


class TestFC(unittest.TestCase):
    def test_json(self):
        with open('../io_data/student_exam.json', 'r') as f:
            test_scores = f.read()
        with open('../io_data/student_knowledge_statistics.json', 'r') as f:
            knowledge_scores = f.read()

        actual_scores = handler(test_scores, None)

        self.assertEqual(knowledge_scores, actual_scores)


if __name__ == '__main__':
    unittest.main()
