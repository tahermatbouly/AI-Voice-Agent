import asyncio

from Backend.agent.state import AgentState
from Backend.agent.extraction import extract_candidate_info


async def main():

    state: AgentState = {
        "transcript": (
            "أنا أحمد محمد، عندي 3 سنين خبرة "
            "في البرمجة"
        ),

        "candidate": {},

        "questions": [],

        "current_question_index": 1,

        "current_question": {
            "id": "q1_name",
            "question_ar": "ممكن أعرف اسم حضرتك بالكامل؟",
            "target_field": "candidate_name",
        },

        "response": "",

        "interview_finished": False,
    }

    result = await extract_candidate_info(state)

    print()
    print("=" * 60)
    print("CANDIDATE")
    print("=" * 60)
    print(result["candidate"])
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())