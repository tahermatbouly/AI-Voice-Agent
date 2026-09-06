
from __future__ import annotations

from typing import Optional

from livekit.agents import RunContext, function_tool


# ============================================================
# FIELDS
# ============================================================

EXTRACTION_FIELDS = (
    "candidate_name",
    "target_domain",
    "years_of_experience",
    "education_level",
    "key_skills_and_technologies",
    "english_proficiency",
)


# ============================================================
# CALL EXTRACTION
# ============================================================

class CallExtraction:
    """Stores structured information for the current call."""

    def __init__(self) -> None:
        self.record: dict[str, str] = {
            field: ""
            for field in EXTRACTION_FIELDS
        }

    def update(
        self,
        candidate_name: Optional[str] = None,
        target_domain: Optional[str] = None,
        years_of_experience: Optional[str] = None,
        education_level: Optional[str] = None,
        key_skills_and_technologies: Optional[str] = None,
        english_proficiency: Optional[str] = None,
    ) -> None:

        values = {
            "candidate_name": candidate_name,
            "target_domain": target_domain,
            "years_of_experience": years_of_experience,
            "education_level": education_level,
            "key_skills_and_technologies": key_skills_and_technologies,
            "english_proficiency": english_proficiency,
        }

        for field, value in values.items():
            if value is None:
                continue

            value = str(value).strip()

            if value:
                self.record[field] = value

    def get_record(self) -> dict[str, str]:
        return self.record.copy()

    def get_missing_fields(self) -> list[str]:
        return [
            field
            for field in EXTRACTION_FIELDS
            if not self.record[field]
        ]

    def has_information(self) -> bool:
        return any(self.record.values())

    def reset(self) -> None:
        for field in EXTRACTION_FIELDS:
            self.record[field] = ""


# ============================================================
# EXTRACTION TOOL
# ============================================================

class ExtractionTools:
    """
    Provides a single-field extraction tool.

    The question manager decides which field is currently being
    collected, so the LLM only needs to provide the value.
    """

    def __init__(self, extraction: CallExtraction) -> None:
        self.extraction = extraction
        self.allowed_fields: set[str] = set()

    def set_allowed_fields(self, fields: list[str]) -> None:
        self.allowed_fields = set(fields)

    @function_tool
    async def update_candidate_info(
        self,
        context: RunContext,
        value: str | None,
    ) -> str:
        """
        Save the candidate's answer to the field requested
        by the current interview question.
        """

        if value is None:
            return "No information was provided."

        value = str(value).strip()

        if not value:
            return "No information was provided."

        for field in self.allowed_fields:
            self.extraction.update(
                **{field: value}
            )

        return "Candidate information updated."
