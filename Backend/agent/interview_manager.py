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

        # ----------------------------------------------------
        # Separate real interview questions from system
        # messages.
        # ----------------------------------------------------

        self.interview_questions = [
            question
            for question in self.questions
            if question.get("type") != "system_message"
        ]

        self.system_messages = {
            question["id"]: question
            for question in self.questions
            if question.get("type") == "system_message"
        }

    # ========================================================
    # NORMAL QUESTIONS
    # ========================================================

    def get_question(self, index: int):

        if index < 0:
            return None

        if index >= len(self.interview_questions):
            return None

        return self.interview_questions[index]

    def get_next_question(self, index: int):

        next_index = index + 1

        if next_index >= len(self.interview_questions):
            return None

        return self.interview_questions[next_index]

    def is_finished(self, index: int):

        return index >= len(self.interview_questions)

    # ========================================================
    # SYSTEM MESSAGES
    # ========================================================

    def get_system_message(self, message_id: str):

        return self.system_messages.get(message_id)

    # ========================================================
    # WAITING FOR SUMMARY
    # ========================================================

    def get_waiting_for_summary(self):

        return self.get_system_message(
            "waiting_for_summary"
        )