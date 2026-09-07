import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ConversationStore:

    def __init__(
        self,
        storage_dir: str = "Backend/data/conversations",
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Generate call ID
    # ========================================================

    def create_call_id(self) -> str:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        return f"call_{timestamp}"

    # ========================================================
    # Current timestamp
    # ========================================================

    @staticmethod
    def timestamp() -> str:

        return datetime.now().astimezone().isoformat()

    # ========================================================
    # Create new conversation
    # ========================================================

    def create_call(
        self,
        call_id: str,
        candidate: dict | None = None,
    ) -> dict:

        conversation = {
            "call_id": call_id,
            "started_at": self.timestamp(),
            "ended_at": None,
            "status": "in_progress",
            "candidate": candidate or {},
            "conversation": [],
        }

        self.save_call(
            call_id,
            conversation,
        )

        return conversation

    # ========================================================
    # Add message
    # ========================================================

    def add_message(
        self,
        call: dict,
        role: str,
        text: str,
        question_id: str | None = None,
        accepted: bool | None = None,
        extracted: dict[str, Any] | None = None,
        repeat: bool = False,
    ):

        message = {
            "role": role,
            "question_id": question_id,
            "text": text,
            "timestamp": self.timestamp(),
        }

        if accepted is not None:
            message["accepted"] = accepted

        if extracted:
            message["extracted"] = extracted

        if repeat:
            message["repeat"] = True

        call["conversation"].append(message)

    # ========================================================
    # Update candidate
    # ========================================================

    def update_candidate(
        self,
        call: dict,
        candidate: dict,
    ):

        call["candidate"] = dict(candidate)

    # ========================================================
    # Save
    # ========================================================

    def save_call(
        self,
        call_id: str,
        call: dict,
    ):

        path = self.storage_dir / f"{call_id}.json"

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                call,
                file,
                ensure_ascii=False,
                indent=2,
            )

    # ========================================================
    # Complete call
    # ========================================================

    def complete_call(
        self,
        call: dict,
        status: str = "completed",
    ):

        call["ended_at"] = self.timestamp()
        call["status"] = status

        self.update_candidate(
            call,
            call.get("candidate", {}),
        )

        self.save_call(
            call["call_id"],
            call,
        )