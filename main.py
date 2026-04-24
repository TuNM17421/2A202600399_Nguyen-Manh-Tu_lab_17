"""
main.py — Interactive CLI for the MemoryAgent.
Usage: python main.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.agent.memory_agent import MemoryAgent


def main():
    print("=" * 60)
    print("Lab 17 — Multi-Memory Agent")
    print("Type 'exit' to quit | 'status' to see token usage")
    print("=" * 60)

    session_id = input("Session ID (press Enter for 'default'): ").strip() or "default"
    agent = MemoryAgent(session_id=session_id)

    print(f"\nSession '{session_id}' started. Redis ping: {agent.redis.ping()}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "status":
            print(f"\n[Memory type used: {agent.last_memory_type}]")
            print(agent.last_token_summary)
            print()
            continue

        response = agent.chat(user_input)
        print(f"\nAgent [{agent.last_memory_type}]: {response}\n")


if __name__ == "__main__":
    main()
