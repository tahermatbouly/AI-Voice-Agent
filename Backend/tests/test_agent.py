import asyncio

from Backend.agent.graph import agent_graph
from Backend.agent.interview_manager import InterviewManager


QUESTIONS_PATH = "Backend/agent/questions.json"


async def main():

    manager = InterviewManager(
        QUESTIONS_PATH
    )

    first_question = manager.get_question(1)

    state = {
        "transcript": (
            "أنا أحمد محمد، عندي 3 سنين خبرة "
            "في البرمجة"
        ),

        "candidate": {},

        "questions": manager.questions,

        "current_question_index": 1,

        "current_question": first_question,

        "response": first_question["question_ar"],

        "interview_finished": False,
    }

    print()
    print("=" * 60)
    print("CURRENT QUESTION")
    print("=" * 60)

    print(
        state["current_question"]["question_ar"]
    )

    print()
    print("=" * 60)
    print("CANDIDATE ANSWER")
    print("=" * 60)

    print(state["transcript"])

    result = await agent_graph.ainvoke(
        state
    )

    print()
    print("=" * 60)
    print("EXTRACTED CANDIDATE")
    print("=" * 60)

    print(result["candidate"])

    print()
    print("=" * 60)
    print("NEXT QUESTION")
    print("=" * 60)

    print(
        result["current_question"]["id"]
    )

    print(
        result["response"]
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())