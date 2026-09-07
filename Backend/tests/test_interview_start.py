import asyncio

from Backend.agent.graph import agent_graph


async def main():

    state = {
        "transcript": "",

        "candidate": {},

        "questions": [],

        "current_question_index": -1,

        "current_question": None,

        "response": "",

        "interview_finished": False,
    }

    result = await agent_graph.ainvoke(
        state
    )

    print()
    print("=" * 60)
    print("INTERVIEW START")
    print("=" * 60)

    print(
        "Question ID:",
        result["current_question"]["id"],
    )

    print(
        "Question:"
    )

    print(
        result["response"]
    )

    print(
        "Finished:",
        result["interview_finished"],
    )

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())