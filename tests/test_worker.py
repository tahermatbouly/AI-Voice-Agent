from agent.extraction import CallExtraction, ExtractionTools
from agent.worker import RecruitmentAgent


def main():
    print("=" * 70)
    print("WORKER / AGENT CONSTRUCTION TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Create extraction state
    # ---------------------------------------------------------------
    print("\n[1] Creating CallExtraction...")

    extraction = CallExtraction()

    print("OK")
    print("Initial record:")
    print(extraction.get_record())

    # ---------------------------------------------------------------
    # 2. Create extraction tools
    # ---------------------------------------------------------------
    print("\n[2] Creating ExtractionTools...")

    tools = ExtractionTools(extraction)

    print("OK")

    # ---------------------------------------------------------------
    # 3. Inspect extraction tool
    # ---------------------------------------------------------------
    print("\n[3] Checking extraction tool...")

    tool = tools.update_candidate_info

    print(f"Tool type: {type(tool)}")
    print(f"Tool ID: {tool.id}")
    print(f"Tool info: {tool.info}")

    # ---------------------------------------------------------------
    # 4. Create RecruitmentAgent
    # ---------------------------------------------------------------
    print("\n[4] Creating RecruitmentAgent...")

    agent = RecruitmentAgent(extraction)

    print("OK")

    # ---------------------------------------------------------------
    # 5. Verify registered tools
    # ---------------------------------------------------------------
    print("\n[5] Checking registered agent tools...")

    print(f"Number of tools: {len(agent._tools)}")

    for index, registered_tool in enumerate(agent._tools, start=1):
        print(f"\nTool {index}:")
        print(f"  Type: {type(registered_tool)}")
        print(f"  ID: {registered_tool.id}")
        print(f"  Info: {registered_tool.info}")

    # ---------------------------------------------------------------
    # 6. Verify extraction tool is actually registered
    # ---------------------------------------------------------------
    assert len(agent._tools) == 1

    registered_tool = agent._tools[0]

    assert registered_tool.id == tool.id

    print("\n[6] Tool registration check...")
    print("OK")

    # ---------------------------------------------------------------
    # 7. Verify system prompt
    # ---------------------------------------------------------------
    print("\n[7] Checking agent instructions...")

    assert agent.instructions == agent.instructions
    assert len(agent.instructions) > 0

    print("OK")
    print(f"Instructions length: {len(agent.instructions)} characters")

    # ---------------------------------------------------------------
    # Final
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WORKER / AGENT CONSTRUCTION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()