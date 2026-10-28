import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tasks.task_output import TaskOutput
from crewai_tools import SerperDevTool


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY is missing from .env")

TODAY = datetime.now(
    ZoneInfo("Asia/Kolkata")
).strftime("%Y-%m-%d")

llm = LLM(
    model="openai/gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.2
)

# Used only by agents that need web search.
search_tool = SerperDevTool()


# ============================================================
# 2. AGENT 1: RESEARCHER
# ============================================================

researcher = Agent(
    role="Senior Technology News Researcher",

    goal=(
        f"Discover today's ({TODAY}) top 3 important technology "
        "news stories using current web search results. "
        "Prioritize recency, credibility, importance, and diversity."
    ),

    backstory=(
        "You are an experienced technology journalist. "
        "You research AI, software engineering, cybersecurity, "
        "cloud computing, semiconductors, and major technology companies. "
        "You distinguish publication dates from event dates, "
        "avoid duplicate stories, and never invent sources or facts."
    ),

    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 3. AGENT 2: FACT CHECKER
# ============================================================

fact_checker = Agent(
    role="Technology News Fact Checker",

    goal=(
        "Independently verify the researcher's candidate stories, "
        "check source credibility, confirm publication dates, "
        "identify duplicate stories, and reject unsupported claims."
    ),

    backstory=(
        "You are a meticulous fact-checking journalist. "
        "You use web search to examine original sources and credible "
        "reporting. You distinguish confirmed facts from allegations, "
        "opinions, and speculation. You never claim to have verified "
        "an article if its contents were not accessible."
    ),

    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 4. AGENT 3: NEWSLETTER WRITER
# ============================================================

writer = Agent(
    role="Technology Newsletter Writer",

    goal=(
        "Write a concise, engaging newsletter based exclusively "
        "on the verified research supplied through task context."
    ),

    backstory=(
        "You are a professional technology newsletter editor. "
        "You turn complex technical developments into clear, "
        "readable summaries for a general audience. "
        "You never perform web searches or introduce unsupported facts."
    ),

    # The Writer must not have any tools.
    tools=[],
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 5. AGENT 4: QUALITY REVIEWER
# ============================================================

reviewer = Agent(
    role="Newsletter Quality Reviewer",

    goal=(
        "Review the newsletter for factual consistency, readability, "
        "three distinct stories, and compliance with the exact "
        "five-line output requirement."
    ),

    backstory=(
        "You are a strict editorial quality reviewer. "
        "You compare the newsletter against the verified research "
        "provided in context. You correct unsupported statements, "
        "duplicate stories, formatting errors, and unnecessary wording. "
        "You do not conduct independent web searches."
    ),

    tools=[],
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 6. TASK 1: RESEARCH TODAY'S TOP 3 TECH STORIES
# ============================================================

research_task = Task(
    description=(
        f"Today's date is {TODAY}, in the Asia/Kolkata timezone.\n\n"

        "Research today's top 3 technology news stories.\n\n"

        "Instructions:\n"
        "1. Search multiple technology news categories.\n"
        "2. Gather several candidate stories before selecting the best three.\n"
        "3. Prefer significant developments published on today's date.\n"
        "4. Check the publication date instead of relying on search-result order.\n"
        "5. Prefer original announcements and credible news publications.\n"
        "6. Avoid duplicate coverage of the same underlying event.\n"
        "7. Provide a concise factual summary for every selected story.\n"
        "8. Include headline, publication date, publisher, and source URL.\n"
        "9. Clearly flag dates or article contents that could not be verified.\n"
        "10. Never fabricate news, dates, quotes, or URLs.\n\n"

        "Return three distinct candidate stories with enough evidence "
        "for the Fact Checker to investigate each one."
    ),

    expected_output=(
        "Three distinct candidate technology news stories. "
        "Each includes a headline, summary, publication date, "
        "publisher, source URL, and any verification limitations."
    ),

    agent=researcher
)


# ============================================================
# 7. TASK 2: FACT-CHECK THE STORIES
# ============================================================

fact_check_task = Task(
    description=(
        f"Independently fact-check the researcher's findings for {TODAY}.\n\n"

        "Use the preceding research task as your candidate list.\n\n"

        "For each story:\n"
        "1. Search for the original source or credible corroborating sources.\n"
        "2. Verify the publication date whenever the source permits.\n"
        "3. Check that the headline and summary match the available evidence.\n"
        "4. Confirm that the source URL refers to the relevant story.\n"
        "5. Identify duplicates, unsupported claims, and misleading wording.\n"
        "6. Reject stories that cannot meet the required evidence standard.\n"
        "7. Never claim to have opened or read an inaccessible article.\n\n"

        "Return the best three verified stories if three qualify. "
        "If fewer than three qualify, explicitly report the shortfall. "
        "Do not invent replacement stories to reach three."
    ),

    expected_output=(
        "A fact-check report containing verified stories, their headlines, "
        "accurate summaries, publication dates, publishers, source URLs, "
        "and verification notes. Identify any stories rejected or "
        "any shortfall in the number of verified stories."
    ),

    agent=fact_checker,

    # Explicitly pass the Researcher's result to the Fact Checker.
    context=[research_task]
)


# ============================================================
# 8. TASK 3: WRITE THE NEWSLETTER
# ============================================================

newsletter_task = Task(
    description=(
        "Write a technology newsletter using ONLY the Fact Checker's "
        "report supplied through task context.\n\n"

        "Output requirements:\n"
        "Line 1: An engaging newsletter title.\n"
        "Line 2: The first verified technology news story.\n"
        "Line 3: The second verified technology news story.\n"
        "Line 4: The third verified technology news story.\n"
        "Line 5: A brief closing sentence.\n\n"

        "Additional rules:\n"
        "- Output exactly five lines.\n"
        "- Do not include blank lines.\n"
        "- Do not add introductory commentary or a conclusion on another line.\n"
        "- Do not add markdown code fences.\n"
        "- Keep every line concise.\n"
        "- Do not invent facts or add unsupported claims.\n"
        "- Do not perform web searches.\n"
        "- If the Fact Checker did not verify three stories, do not fabricate "
        "a third story. Report the verification shortfall instead."
    ),

    expected_output=(
        "Exactly five non-empty lines: a title, three verified news stories, "
        "and a closing sentence. No blank lines, commentary, or code fences. "
        "If fewer than three stories were verified, clearly report the "
        "shortfall rather than inventing news."
    ),

    agent=writer,

    # The Writer receives the Fact Checker's output, not just the
    # original Researcher's output.
    context=[fact_check_task]
)


# ============================================================
# 9. PYTHON VALIDATOR FOR THE FINAL NEWSLETTER
# ============================================================

def validate_newsletter(result: TaskOutput):
    """
    CrewAI task guardrail.

    Enforces exactly five non-empty lines.
    Checks the expected title, story, and closing-line structure.
    """

    text = result.raw.strip()
    lines = text.splitlines()

    if len(lines) != 5:
        return False, (
            f"Expected exactly 5 lines but received {len(lines)}. "
            "Rewrite the output as exactly five non-empty lines."
        )

    if any(not line.strip() for line in lines):
        return False, (
            "Blank lines are not allowed. "
            "Return exactly five non-empty lines."
        )

    # Require a recognizable title on line 1.
    if len(lines[0].strip()) < 5:
        return False, "Line 1 must contain a meaningful newsletter title."

    # Require content on all three story lines.
    for index in (1, 2, 3):
        if len(lines[index].strip()) < 15:
            return False, (
                f"Line {index + 1} is too short to be a useful news summary."
            )

    # Require a meaningful closing line.
    if len(lines[4].strip()) < 10:
        return False, "Line 5 must contain a meaningful closing sentence."

    return True, text


# ============================================================
# 10. TASK 4: REVIEW AND VALIDATE THE NEWSLETTER
# ============================================================

review_task = Task(
    description=(
        "Review the Writer's newsletter using both the Writer's output "
        "and the Fact Checker's report in the supplied context.\n\n"

        "Check the following:\n"
        "1. There are exactly three distinct news stories.\n"
        "2. Each story is supported by the verified research.\n"
        "3. No unsupported claims or invented details were introduced.\n"
        "4. The language is clear, concise, and appropriate for general readers.\n"
        "5. The newsletter has exactly five non-empty lines.\n"
        "6. Line 1 is the title; lines 2-4 are stories; line 5 is the closing.\n\n"

        "Correct any issues and return only the final newsletter. "
        "Do not include review notes, explanations, or markdown code fences. "
        "If the research contains fewer than three verified stories, "
        "do not invent missing stories; clearly indicate the shortfall."
    ),

    expected_output=(
        "The reviewed newsletter containing exactly five non-empty lines, "
        "with three evidence-supported stories, a title, and a closing line. "
        "Do not fabricate stories if verification was insufficient."
    ),

    agent=reviewer,

    # The Reviewer sees both the original newsletter and verified research.
    context=[fact_check_task, newsletter_task],

    # Python checks the output and gives feedback for correction.
    guardrail=validate_newsletter,
    guardrail_max_retries=2
)


# ============================================================
# 11. CREATE THE CREW
# ============================================================

news_crew = Crew(
    agents=[
        researcher,
        fact_checker,
        writer,
        reviewer
    ],

    tasks=[
        research_task,
        fact_check_task,
        newsletter_task,
        review_task
    ],

    process=Process.sequential,
    verbose=True
)


# ============================================================
# 12. EXECUTE THE CREW
# ============================================================

def main():
    print("=" * 65)
    print("AI TECHNOLOGY NEWSLETTER CREW")
    print(f"Research date: {TODAY}")
    print("=" * 65)

    try:
        result = news_crew.kickoff()

        final_newsletter = result.raw.strip()
        lines = final_newsletter.splitlines()

        # Final safety check before printing the result.
        if len(lines) != 5 or any(not line.strip() for line in lines):
            raise RuntimeError(
                "The Reviewer did not produce a valid five-line newsletter. "
                "The result will not be published."
            )

        print("\n" + "=" * 65)
        print("FINAL NEWSLETTER")
        print("=" * 65)

        # Print only the five-line newsletter.
        print(final_newsletter)

        print("\n" + "=" * 65)
        print("Validation successful: exactly five non-empty lines.")
        print("=" * 65)

    except Exception as exc:
        print(
            f"\nNewsletter generation failed: {exc}",
            file=sys.stderr
        )
        raise


if __name__ == "__main__":
    main()