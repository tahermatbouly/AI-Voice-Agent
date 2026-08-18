"""
Database layer: SQLite via SQLAlchemy. Stores the final extracted HR
record, full transcript, and the raw call audio for every completed
call, and serves the dashboard's list/detail views.

Note on audio storage: raw audio bytes are stored in the database (a
BLOB column) per project decision, rather than as separate files on
disk. This is simpler operationally (one file, no orphaned audio files
to manage) at the cost of a larger database file over time -- worth
revisiting if call volume grows large enough for this to matter.

Audio is deliberately NOT included in list_calls() or get_call()'s
JSON output -- raw bytes don't belong in a JSON response. Use
get_call_audio() separately (exposed as its own binary-response API
endpoint) to actually fetch the audio for playback/download.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker

from app import config

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    caller_id = Column(String, default="")  # phone number or session identifier, used to link to Mem0 memory

    # HR extraction fields -- must stay in sync with config.EXTRACTION_FIELDS
    candidate_name = Column(String, default="")
    contact_info = Column(String, default="")
    position = Column(String, default="")
    experience = Column(Text, default="")
    current_salary = Column(String, default="")
    expected_salary = Column(String, default="")
    availability = Column(String, default="")
    notes = Column(Text, default="")

    transcript_json = Column(Text, default="[]")  # stored as a JSON string

    # Raw call audio, stored directly in the database.
    audio_data = Column(LargeBinary, nullable=True)
    audio_format = Column(String, default="wav")  # e.g. "wav" or "mp3" -- needed to serve it correctly later


def init_db():
    """Create the calls table if it doesn't exist yet. Call this once at startup."""
    Base.metadata.create_all(bind=engine)


def save_call(
    record: dict,
    transcript: list,
    caller_id: str = "",
    audio_bytes: bytes | None = None,
    audio_format: str = "wav",
) -> int:
    """Save a completed call: extracted record, transcript, and optionally
    the raw call audio. Returns the new call's id."""
    session = SessionLocal()
    try:
        call = Call(
            caller_id=caller_id,
            candidate_name=record.get("candidate_name", ""),
            contact_info=record.get("contact_info", ""),
            position=record.get("position", ""),
            experience=record.get("experience", ""),
            current_salary=record.get("current_salary", ""),
            expected_salary=record.get("expected_salary", ""),
            availability=record.get("availability", ""),
            notes=record.get("notes", ""),
            transcript_json=json.dumps(transcript, ensure_ascii=False),
            audio_data=audio_bytes,
            audio_format=audio_format,
        )
        session.add(call)
        session.commit()
        session.refresh(call)
        return call.id
    finally:
        session.close()


def list_calls() -> list[dict]:
    """Summary of every call, most recent first -- for the dashboard's list view.
    Deliberately excludes audio and the full transcript to keep this light."""
    session = SessionLocal()
    try:
        calls = session.query(Call).order_by(Call.timestamp.desc()).all()
        return [
            {
                "id": c.id,
                "timestamp": c.timestamp.isoformat(),
                "candidate_name": c.candidate_name,
                "position": c.position,
                "has_audio": c.audio_data is not None,
            }
            for c in calls
        ]
    finally:
        session.close()


def get_call(call_id: int) -> dict | None:
    """Full record + transcript for one call -- for the dashboard's detail view.
    Does NOT include raw audio bytes -- use get_call_audio() for that."""
    session = SessionLocal()
    try:
        call = session.query(Call).filter(Call.id == call_id).first()
        if call is None:
            return None
        return {
            "id": call.id,
            "timestamp": call.timestamp.isoformat(),
            "record": {
                "candidate_name": call.candidate_name,
                "contact_info": call.contact_info,
                "position": call.position,
                "experience": call.experience,
                "current_salary": call.current_salary,
                "expected_salary": call.expected_salary,
                "availability": call.availability,
                "notes": call.notes,
            },
            "transcript": json.loads(call.transcript_json),
            "has_audio": call.audio_data is not None,
        }
    finally:
        session.close()


def get_call_audio(call_id: int) -> tuple[bytes, str] | None:
    """Raw audio bytes + format for one call, for a dedicated audio-serving
    endpoint. Returns None if the call doesn't exist or has no audio."""
    session = SessionLocal()
    try:
        call = session.query(Call).filter(Call.id == call_id).first()
        if call is None or call.audio_data is None:
            return None
        return call.audio_data, call.audio_format
    finally:
        session.close()