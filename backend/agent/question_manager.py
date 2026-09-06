from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class QuestionManager:

    def __init__(
        self,
        questions_path: str | Path,
    ) -> None:

        self.questions_path = Path(
            questions_path
        )

        # Load ONCE when the worker/call starts.
        with self.questions_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.questions: list[dict[str, Any]] = json.load(f)

        self._questions_by_id = {
            question["id"]: question
            for question in self.questions
        }

    # ========================================================
    # WELCOME
    # ========================================================

    def get_welcome(self) -> str | None:

        for question in self.questions:

            if question.get("is_welcome"):

                return question["question_ar"]

        return None

    # ========================================================
    # NEXT QUESTION
    # ========================================================

    def get_next_question(
        self,
        candidate_state: dict[str, Any],
    ) -> dict[str, Any] | None:

        for question in self.questions:

            # Skip welcome message
            if question.get("is_welcome"):
                continue

            # ------------------------------------------------
            # Single target field
            # ------------------------------------------------

            target_field = question.get(
                "target_field"
            )

            if target_field:

                if not self._has_value(
                    candidate_state.get(
                        target_field
                    )
                ):
                    return question

                continue

            # ------------------------------------------------
            # Multiple target fields
            # ------------------------------------------------

            target_fields = question.get(
                "target_fields"
            )

            if target_fields:

                missing = any(
                    not self._has_value(
                        candidate_state.get(field)
                    )
                    for field in target_fields
                )

                if missing:
                    return question

        return None

    # ========================================================
    # QUESTION LOOKUP
    # ========================================================

    def get_question(
        self,
        question_id: str,
    ) -> dict[str, Any] | None:

        return self._questions_by_id.get(
            question_id
        )

    # ========================================================
    # VALUE CHECK
    # ========================================================


    # ========================================================
    # TARGET FIELDS
    # ========================================================

    @staticmethod
    def get_target_fields(
        question: dict[str, Any],
    ) -> list[str]:

        target_field = question.get(
            "target_field"
        )

        if target_field:
            return [target_field]

        target_fields = question.get(
            "target_fields"
        )

        if target_fields:
            return list(target_fields)

        return []

    @staticmethod
    def _has_value(
        value: Any,
    ) -> bool:

        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        if isinstance(value, (list, tuple, set)):
            return len(value) > 0

        return True