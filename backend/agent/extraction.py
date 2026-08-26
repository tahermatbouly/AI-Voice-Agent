"""
Structured HR information extraction for the AI voice agent.

This module manages information extracted during a single call.

Mandatory information:
- candidate_name
- position
- experience
- current_salary
- availability
- notes

Not collected:
- contact_info
- expected_salary
"""

from typing import Optional

from livekit.agents import RunContext, function_tool

from app import config


class CallExtraction:
    """
    Holds structured information extracted during one call.

    A new instance must be created for every call.
    """

    def __init__(self):
        self.record = {
            field: "" for field in config.EXTRACTION_FIELDS
        }

    def update(
        self,
        candidate_name: Optional[str] = None,
        position: Optional[str] = None,
        experience: Optional[str] = None,
        current_salary: Optional[str] = None,
        availability: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Update information explicitly provided by the caller.

        Empty and None values are ignored.
        """

        updates = {
            "candidate_name": candidate_name,
            "position": position,
            "experience": experience,
            "current_salary": current_salary,
            "availability": availability,
            "notes": notes,
        }

        for field, value in updates.items():
            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            self.record[field] = value

    def get_record(self) -> dict:
        """Return the currently extracted information."""

        return self.record.copy()

    def get_missing_fields(self) -> list[str]:
        """Return mandatory fields that are still missing."""

        return [
            field
            for field in config.EXTRACTION_FIELDS
            if not self.record.get(field)
        ]

    def has_information(self) -> bool:
        """Return True if at least one field has been extracted."""

        return any(
            bool(value)
            for value in self.record.values()
        )


class ExtractionTools:
    """
    Function tools exposed to the conversational LLM.

    The LLM calls this tool when the caller provides HR information.
    """

    def __init__(self, extraction: CallExtraction):
        self.extraction = extraction

    @function_tool
    async def update_candidate_info(
        self,
        context: RunContext,
        candidate_name: str | None = None,
        position: str | None = None,
        experience: str | None = None,
        current_salary: str | None = None,
        availability: str | None = None,
        notes: str | None = None,
    ) -> str:
        """
        Save information explicitly provided by the caller.

        Rules:
        - Only save information the caller actually stated.
        - Never guess or invent information.
        - Only send fields relevant to the latest caller message.
        - Multiple fields can be saved if the caller provides them together.
        - Existing information can be replaced when the caller clearly
          corrects or updates it.
        """

        self.extraction.update(
            candidate_name=candidate_name,
            position=position,
            experience=experience,
            current_salary=current_salary,
            availability=availability,
            notes=notes,
        )

        return "Candidate information updated."