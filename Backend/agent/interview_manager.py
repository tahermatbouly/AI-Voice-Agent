import json
from pathlib import Path


class InterviewManager:

    def __init__(self, questions_path: str):

        self.questions_path = Path(questions_path)

        with open(
            self.questions_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.questions = json.load(file)

    def get_question(self, index: int):

        if index >= len(self.questions):
            return None

        return self.questions[index]

    def is_finished(self, index: int):

        return index >= len(self.questions)

    def get_next_question(self, index: int):

        next_index = index + 1

        if next_index >= len(self.questions):
            return None

        return self.questions[next_index]