
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from agent.audio import AudioManager
from agent.extraction import CallExtraction
from agent.question_manager import QuestionManager
from agent.stt_plugin import CohereArabicSTT
from agent.tts_plugin import VoiceTut


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("voice-agent.worker")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_PATH = BASE_DIR / "questions.json"


# ============================================================
# INTERVIEW WORKER
# ============================================================

class InterviewWorker:

    def __init__(self) -> None:

        logger.info("========================================")
        logger.info("[WORKER] Initializing")
        logger.info("========================================")

        # ----------------------------------------------------
        # Questions
        # ----------------------------------------------------

        logger.info(
            "[WORKER] Questions file: %s",
            QUESTIONS_PATH,
        )

        self.question_manager = QuestionManager(
            QUESTIONS_PATH
        )

        # ----------------------------------------------------
        # Deterministic extraction
        # ----------------------------------------------------

        self.extraction = CallExtraction()

        # ----------------------------------------------------
        # STT
        # ----------------------------------------------------

        self.stt = CohereArabicSTT()

        # ----------------------------------------------------
        # TTS
        # ----------------------------------------------------

        self.tts = VoiceTut()

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        self.audio = AudioManager()

        logger.info("[WORKER] Initialization complete.")

    # ========================================================
    # SPEAK
    # ========================================================

    async def speak(self, text: str) -> None:

        if not text:
            return

        logger.info(
            "[TTS] Speaking: %s",
            text,
        )

        pcm, sample_rate = await self.tts.synthesize(
            text
        )

        if not pcm:
            logger.warning(
                "[TTS] Empty audio returned."
            )
            return

        await self.audio.play(
            pcm=pcm,
            sample_rate=sample_rate,
            channels=1,
        )

    # ========================================================
    # PROCESS ANSWER
    # ========================================================

    async def process_answer(
        self,
        question: dict,
        transcript: str,
    ) -> None:

        transcript = transcript.strip()

        if not transcript:

            logger.warning(
                "[EXTRACTION] Empty transcript."
            )

            return

        target_field = question.get(
            "target_field"
        )

        if not target_field:

            logger.warning(
                "[EXTRACTION] Question %s has no target_field.",
                question.get("id"),
            )

            return

        logger.info(
            "[EXTRACTION] Saving %s = %s",
            target_field,
            transcript,
        )

        # ----------------------------------------------------
        # Deterministic extraction.
        #
        # The current question tells us exactly which
        # field the transcript belongs to.
        # ----------------------------------------------------

        self.extraction.update(
            **{
                target_field: transcript
            }
        )

        logger.info(
            "[EXTRACTION] Current record: %s",
            self.extraction.get_record(),
        )

    # ========================================================
    # RUN QUESTION
    # ========================================================

    async def run_question(
        self,
        question: dict,
    ) -> None:

        question_id = question.get(
            "id",
            "<unknown>",
        )

        question_text = question.get(
            "question_ar",
            "",
        )

        logger.info("----------------------------------------")
        logger.info(
            "[QUESTION] %s",
            question_id,
        )
        logger.info(
            "[QUESTION] %s",
            question_text,
        )
        logger.info("----------------------------------------")

        # ----------------------------------------------------
        # 1. TTS
        # ----------------------------------------------------

        await self.speak(
            question_text
        )

        # ----------------------------------------------------
        # 2. Record answer
        # ----------------------------------------------------

        # IMPORTANT:
        #
        # We intentionally do NOT use:
        # - VAD
        # - turn detection
        # - interruption detection
        #
        # For now, recording has a fixed duration.
        # ----------------------------------------------------

        answer_duration = 8.0

        logger.info(
            "[AUDIO] Recording answer for %.1f seconds...",
            answer_duration,
        )

        pcm = await self.audio.record(
            duration=answer_duration
        )

        # ----------------------------------------------------
        # 3. STT
        # ----------------------------------------------------

        transcript = await self.stt.transcribe(
            pcm=pcm,
            sample_rate=self.audio.sample_rate,
            num_channels=self.audio.channels,
        )

        logger.info(
            "[STT] Transcript: %s",
            transcript if transcript else "<EMPTY>",
        )

        # ----------------------------------------------------
        # 4. Extraction
        # ----------------------------------------------------

        await self.process_answer(
            question=question,
            transcript=transcript,
        )

    # ========================================================
    # RUN INTERVIEW
    # ========================================================

    async def run(self) -> None:

        logger.info("========================================")
        logger.info("[INTERVIEW] Starting interview")
        logger.info("========================================")

        # ----------------------------------------------------
        # Welcome
        # ----------------------------------------------------

        welcome = (
            self.question_manager.get_welcome()
        )

        if welcome:

            await self.speak(
                welcome
            )

        # ----------------------------------------------------
        # Sequential interview
        # ----------------------------------------------------

        while True:

            candidate_state = (
                self.extraction.get_record()
            )

            question = (
                self.question_manager.get_next_question(
                    candidate_state
                )
            )

            # ------------------------------------------------
            # Finished
            # ------------------------------------------------

            if question is None:

                logger.info(
                    "[INTERVIEW] All required fields collected."
                )

                break

            # ------------------------------------------------
            # Question → Answer → STT → Extraction
            # ------------------------------------------------

            await self.run_question(
                question
            )

        # ----------------------------------------------------
        # Closing
        # ----------------------------------------------------

        await self.speak(
            "تمام، شكراً ليك. كده خلصنا."
        )

        # ----------------------------------------------------
        # Final record
        # ----------------------------------------------------

        logger.info("========================================")
        logger.info("[INTERVIEW] FINAL RECORD")
        logger.info(
            "%s",
            self.extraction.get_record(),
        )
        logger.info("========================================")

        return self.extraction.get_record()

    # ========================================================
    # SHUTDOWN
    # ========================================================

    async def shutdown(self) -> None:

        logger.info(
            "[WORKER] Shutting down..."
        )

        try:

            await self.stt.aclose()

        except Exception:

            logger.exception(
                "[WORKER] Failed to close STT."
            )

        logger.info(
            "[WORKER] Shutdown complete."
        )


# ============================================================
# MAIN
# ============================================================

async def main() -> None:

    worker = InterviewWorker()

    try:

        await worker.run()

    except KeyboardInterrupt:

        logger.info(
            "[WORKER] Interrupted by user."
        )

    except Exception:

        logger.exception(
            "[WORKER] Fatal error."
        )

        raise

    finally:

        await worker.shutdown()


if __name__ == "__main__":

    asyncio.run(main())
