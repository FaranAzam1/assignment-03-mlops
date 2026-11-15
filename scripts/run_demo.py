"""Run the five assignment scenarios against the live stack and save transcripts to report/evidence/.

    python scripts/run_demo.py            # all five
    python scripts/run_demo.py 3 5        # selected scenarios

Scenario 3 triggers the training DAG; the script asks for confirmation on the console, like the CLI.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from agent.llm import OllamaClient  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "report" / "evidence"

SCENARIOS = {
    1: ("dataset_provenance", "Which dataset version was used to train the best model run, and how can I reproduce it?"),
    2: ("model_selection", "Compare the last five training runs and recommend the best candidate based on F1 score."),
    3: ("pipeline_execution", "Trigger the model training pipeline and report whether execution succeeded."),
    4: ("deployment_diagnosis", "Why is my prediction API unavailable?"),
    5: (
        "multi_tool_reasoning",
        "Find the best recent model, identify the dataset version used to train it, check the deployment "
        "status, and explain whether retraining is necessary.",
    ),
}


def render(question: str, resp) -> str:
    lines = [f"USER: {question}", "", f"[request {resp.request_id} | mode {resp.mode}]", "",
             "TOOL CALLS:"]
    lines += [f"  - {t.tool} {t.args} -> {t.status} ({t.duration_ms:.0f} ms)" for t in resp.tool_calls]
    lines += ["", "AGENT:", resp.answer]
    return "\n".join(lines)


def main() -> None:
    wanted = [int(a) for a in sys.argv[1:]] or list(SCENARIOS)
    OUT.mkdir(parents=True, exist_ok=True)
    agent = Agent(llm=OllamaClient())
    for n in wanted:
        slug, question = SCENARIOS[n]
        resp = agent.handle(question)
        text = render(question, resp)
        for action in resp.pending_actions:
            print(f"\n{action.description}")
            if input("Confirm this state-changing action? [y/N] ").strip().lower() == "y":
                done = agent.confirm(action.action_id)
                text += "\n\nUSER CONFIRMED -> " + done.answer
                time.sleep(100)  # let the DAG run finish, then report its real status
                follow = agent.handle("What is the status of the latest airflow training pipeline runs?")
                text += "\n\n" + render("What is the status of the latest airflow training pipeline runs?", follow)
        (OUT / f"scenario_{n}_{slug}.txt").write_text(text, encoding="utf-8")
        print(f"\n===== scenario {n}: {slug} =====\n{text}\n")


if __name__ == "__main__":
    main()
