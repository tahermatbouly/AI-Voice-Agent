from Backend.agent.interview_manager import InterviewManager


def main():

    manager = InterviewManager(
        "Backend/agent/questions.json"
    )

    print()
    print("=" * 60)
    print("INTERVIEW")
    print("=" * 60)

    while not manager.is_finished():

        question = manager.get_current_question()

        print(
            f"{question['id']}: "
            f"{question['question_ar']}"
        )

        manager.advance()

    print("=" * 60)
    print("INTERVIEW FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()