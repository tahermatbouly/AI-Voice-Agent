import json
from pathlib import Path


# Ids for the non-field spoken items. Kept here (not in the JSON)
# because the code refers to them directly — the JSON only supplies
# their text.
INTRO_ID = "intro"
COMPLETION_ID = "completion"
BEFORE_FOLLOWUPS_ID = "before_followups"
NO_SPEECH_RETRY_ID = "no_speech_retry"
WRONG_ANSWER_RETRY_ID = "wrong_answer_retry"


def field_prompt_id(field_name: str) -> str:
    """
    The TTS cache id for a single field's follow-up question.
    Derived from the field name so it's stable across runs and
    never collides with intro/completion/system message ids.
    """
    return f"ask_{field_name}"


class InterviewManager:
    """
    Loads the one-paragraph interview config.

    Everything spoken by the agent is a static item with a stable
    id, EXCEPT nothing — in this mode there is no dynamically built
    text at all, so every single clip can be pre-generated and
    cached once, ever. (The old per-candidate summary paragraph was
    the only thing that couldn't be, and it's gone.)
    """

    def __init__(self, questions_path: str):

        self.questions_path = Path(questions_path)

        with open(
            self.questions_path,
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(file)

        self.mode = config.get("mode", "one_paragraph")

        self.intro_text = config["intro"]["text"]

        self.completion_text = config["completion"]["text"]

        self.fields = config["fields"]

        self.system_messages = {
            item["id"]: item["text"]
            for item in config.get("system_messages", [])
        }

        self._fields_by_name = {
            field["name"]: field
            for field in self.fields
        }

    # ========================================================
    # FIELDS
    # ========================================================

    def get_field(self, name: str) -> dict | None:
        return self._fields_by_name.get(name)

    def askable_fields(self) -> list[dict]:
        """
        Fields we can ask a follow-up question about. A field with
        "question": null (e.g. notes) can only ever be filled from
        the opening paragraph — we never ask for it directly.
        """
        return [
            field
            for field in self.fields
            if field.get("question")
        ]

    def missing_required(self, candidate: dict) -> list[dict]:
        """
        Required fields still empty, in the order declared in the
        JSON — that order is the order they'll be asked in.

        Only fields with a question are returned: a required field
        with no question would otherwise deadlock the interview.
        """

        missing = []

        for field in self.fields:

            if not field.get("required"):
                continue

            if not field.get("question"):
                continue

            value = candidate.get(field["name"])

            if value is None:
                missing.append(field)
                continue

            if isinstance(value, (list, str)) and not value:
                missing.append(field)

        return missing

    # ========================================================
    # SYSTEM MESSAGES
    # ========================================================

    def get_system_message(self, message_id: str) -> str | None:
        return self.system_messages.get(message_id)

    # ========================================================
    # TTS PRE-GENERATION
    # ========================================================

    def tts_items(self) -> list[tuple[str, str]]:
        """
        Every (id, text) pair the agent can ever speak, for the
        pre-generation/cache pass in handler.py. Because this mode
        has no dynamic text, this list is exhaustive — after the
        first run against a given speaker, no live VoiceTut call
        ever happens again.
        """

        items = [
            (INTRO_ID, self.intro_text),
            (COMPLETION_ID, self.completion_text),
        ]

        for field in self.askable_fields():
            items.append(
                (
                    field_prompt_id(field["name"]),
                    field["question"],
                )
            )

        for message_id, text in self.system_messages.items():
            items.append((message_id, text))

        return items