"""
Cross-call memory for the AI voice agent.

This module is responsible only for durable information that should
survive between separate calls from the same caller.

Current-call information belongs to extraction.py.
Permanent call records belong to db.py.
Cross-call memory belongs here.
"""

from __future__ import annotations

import logging

from app import config

logger = logging.getLogger(__name__)


class CallerMemory:
    """
    Wrapper around Mem0 for cross-call caller memory.

    Each caller is identified by caller_id.

    Example:
        caller_id = phone number
        caller_id = browser/session identifier

    The rest of the application does not need to know how Mem0 works.
    """

    def __init__(self):
        self.memory = None

        try:
            from mem0 import Memory

            self.memory = Memory()

            logger.info("Mem0 memory initialized successfully.")

        except Exception as exc:
            logger.warning(
                "Mem0 could not be initialized: %s",
                exc,
            )

    # -----------------------------------------------------------------
    # Availability
    # -----------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Return True when Mem0 is initialized and available."""
        return self.memory is not None

    # -----------------------------------------------------------------
    # Retrieve memories
    # -----------------------------------------------------------------

    def get_memories(
        self,
        caller_id: str,
        query: str = "Relevant information about this caller",
        limit: int = 5,
    ) -> list[str]:
        """
        Retrieve relevant memories for a caller.

        Returns a simple list of memory strings so the agent layer does
        not need to understand Mem0's response format.
        """

        if not self.available:
            return []

        if not caller_id:
            return []

        try:
            result = self.memory.search(
                query,
                filters={"user_id": caller_id},
                limit=limit,
            )

            results = result.get("results", [])

            return [
                item["memory"]
                for item in results
                if item.get("memory")
            ]

        except Exception as exc:
            logger.warning(
                "Failed to retrieve memories for caller %s: %s",
                caller_id,
                exc,
            )
            return []

    # -----------------------------------------------------------------
    # Save memories
    # -----------------------------------------------------------------

    def save_conversation(
        self,
        caller_id: str,
        conversation: list[dict],
    ) -> bool:
        """
        Save a completed conversation to the caller's memory.

        Mem0 analyzes the conversation and decides which durable facts
        are worth remembering.

        Returns True when the operation succeeds.
        """

        if not self.available:
            return False

        if not caller_id:
            logger.warning(
                "Cannot save memory without caller_id."
            )
            return False

        if not conversation:
            return False

        try:
            self.memory.add(
                conversation,
                user_id=caller_id,
            )

            logger.info(
                "Conversation saved to Mem0 for caller %s.",
                caller_id,
            )

            return True

        except Exception as exc:
            logger.warning(
                "Failed to save memory for caller %s: %s",
                caller_id,
                exc,
            )
            return False

    # -----------------------------------------------------------------
    # Save structured caller information
    # -----------------------------------------------------------------

    def save_record(
        self,
        caller_id: str,
        record: dict,
    ) -> bool:
        """
        Save structured information from a completed call.

        Only non-empty fields are included.

        This is useful at the end of a call when CallExtraction has
        already produced the structured candidate record.
        """

        if not self.available:
            return False

        if not caller_id:
            logger.warning(
                "Cannot save memory without caller_id."
            )
            return False

        information = {
            field: value
            for field, value in record.items()
            if value not in (None, "")
        }

        if not information:
            return False

        message = {
            "role": "user",
            "content": (
                "Candidate information from a completed HR call:\n"
                + "\n".join(
                    f"{field}: {value}"
                    for field, value in information.items()
                )
            ),
        }

        try:
            self.memory.add(
                [message],
                user_id=caller_id,
            )

            logger.info(
                "Structured caller information saved to Mem0 "
                "for caller %s.",
                caller_id,
            )

            return True

        except Exception as exc:
            logger.warning(
                "Failed to save structured memory for caller %s: %s",
                caller_id,
                exc,
            )
            return False