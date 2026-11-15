"""Command-line chat interface: `python -m agent.main` or `python -m agent.main "question"`."""

from __future__ import annotations

import argparse
import sys

from agent.llm import OllamaClient
from agent.orchestrator import Agent


def build_agent(use_llm: bool = True) -> Agent:
    return Agent(llm=OllamaClient() if use_llm else None)


def ask(agent: Agent, message: str) -> None:
    resp = agent.handle(message)
    print(resp.answer)
    print(f"\n[request {resp.request_id} | mode {resp.mode}]")
    for action in resp.pending_actions:
        print(f"\nThis changes state: {action.description}")
        if input("Confirm? [y/N] ").strip().lower() == "y":
            print(agent.confirm(action.action_id).answer)
        else:
            agent.reject(action.action_id)
            print("Cancelled; nothing was changed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLOps AI agent")
    parser.add_argument("message", nargs="*", help="one-shot question (omit for interactive mode)")
    parser.add_argument("--no-llm", action="store_true", help="use the deterministic planner only")
    args = parser.parse_args(argv)
    agent = build_agent(not args.no_llm)
    if args.message:
        ask(agent, " ".join(args.message))
        return 0
    print("MLOps agent. Type a question, or 'exit'.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if line.lower() in {"exit", "quit"}:
            return 0
        if line:
            ask(agent, line)


if __name__ == "__main__":
    sys.exit(main())
