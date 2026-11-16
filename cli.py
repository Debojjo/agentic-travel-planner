"""Command-line interface.

Examples:
    python cli.py "3-day trip from Bangalore to Goa starting 5 December, budget 25000"
    python cli.py "cheapest hotels in Goa with a pool"       # quick questions work too
    python cli.py "..." --json      # also print the JSON plan
    python cli.py "..." --steps     # show which tools the agent called
"""

import argparse
import logging
import sys

from agent import plan_trip


def main() -> None:
    """Parse arguments, run the planner once and print the result."""
    parser = argparse.ArgumentParser(description="Agentic AI travel planner")
    parser.add_argument("query", nargs="?", help="Describe your trip in plain English")
    parser.add_argument("--json", action="store_true", help="Also print the JSON plan")
    parser.add_argument("--steps", action="store_true", help="Show the agent's tool calls")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)

    query = args.query or input("Describe your trip: ")
    result = plan_trip(query=query)

    if result.status != "ok":
        print(f"[{result.status}] {result.message}")
        sys.exit(1)
    if args.steps:
        print("Agent steps:")
        for number, step in enumerate(result.steps, start=1):
            print(f"  {number}. {step['tool']} {step['args']}")
        print()
    print(result.text)
    if result.kind == "answer":
        from ui_helpers import results_to_tables

        for title, table, note in results_to_tables(result.tool_results):
            print(f"\n{title}")
            print(table.to_string(index=False) if table is not None else note)
        return
    if args.json:
        print("\n" + result.json_text)


if __name__ == "__main__":
    main()
