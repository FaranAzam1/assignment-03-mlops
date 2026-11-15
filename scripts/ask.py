"""Ask the agent one question and save the transcript: python scripts/ask.py <slug> "<question>"."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from agent.llm import OllamaClient  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402
from scripts.run_demo import OUT, render  # noqa: E402


def main() -> None:
    slug, question = sys.argv[1], " ".join(sys.argv[2:])
    resp = Agent(llm=OllamaClient()).handle(question)
    text = render(question, resp)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    (Path(OUT) / f"{slug}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
