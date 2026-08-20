"""
Structured HR information extraction for the AI voice agent.

This module manages information extracted during a single call.

The conversational LLM uses the `update_candidate_info` function tool
whenever the caller explicitly provides relevant HR information.

This module is responsible only for CURRENT-CALL extraction.

- extraction.py -> current call's structured information
- memory.py     -> information remembered across calls
- db.py         -> permanent storage after the call finishes
"""

from typing import Optional

from livekit.agents import RunContext, function_tool

from app import config


class CallExtraction:
    """
    Holds the structured information extracted during one call.

    A new instance must be created for every call. This prevents
    information from one caller leaking into another caller's session.
    """

    def __init__(self):
        self.record = {
            field: "" for field in config.EXTRACTION_FIELDS
        }

    def update(
        self,
        candidate_name: Optional[str] = None,
        contact_info: Optional[str] = None,
        position: Optional[str] = None,
        experience: Optional[str] = None,
        current_salary: Optional[str] = None,
        expected_salary: Optional[str] = None,
        availability: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Update the current call's extracted information.

        Only non-empty values are stored.

        The function deliberately does not overwrite an existing value
        with None or an empty string.
        """

        updates = {
            "candidate_name": candidate_name,
            "contact_info": contact_info,
            "position": position,
            "experience": experience,
            "current_salary": current_salary,
            "expected_salary": expected_salary,
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
        """
        Return a copy of the currently extracted information.

        A copy is returned so callers cannot accidentally modify the
        internal extraction state.
        """

        return self.record.copy()

    def get_missing_fields(self) -> list[str]:
        """
        Return the extraction fields that have not been populated yet.
        """

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

    The LLM can call these tools during the live conversation when
    the caller explicitly provides information.
    """

    def __init__(self, extraction: CallExtraction):
        self.extraction = extraction

    @function_tool
    async def update_candidate_info(
        self,
        context: RunContext,
        candidate_name: Optional[str] = None,
        contact_info: Optional[str] = None,
        position: Optional[str] = None,
        experience: Optional[str] = None,
        current_salary: Optional[str] = None,
        expected_salary: Optional[str] = None,
        availability: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Save information explicitly provided by the caller.

        IMPORTANT:
        - Only save information the caller actually stated.
        - Never guess missing information.
        - Do not invent values.
        - Only provide fields relevant to the latest caller message.
        - If multiple fields were provided in one sentence, save them
          together.
        - Existing information should only be replaced when the caller
          clearly provides corrected or updated information.

        Examples:

        Caller:
            "أنا أحمد محمد."

        Tool call:
            candidate_name="أحمد محمد"

        Caller:
            "أنا بقدم على وظيفة مهندس صيانة وعندي 3 سنين خبرة."

        Tool call:
            position="مهندس صيانة"
            experience="3 سنين"

        Caller:
            "مرتبي الحالي 15000 ومتوقع 20000."

        Tool call:
            current_salary="15000"
            expected_salary="20000"
        """

        self.extraction.update(
            candidate_name=candidate_name,
            contact_info=contact_info,
            position=position,
            experience=experience,
            current_salary=current_salary,
            expected_salary=expected_salary,
            availability=availability,
            notes=notes,
        )

        return "Data is saved successfully."