"""Budgeted consolidation and deletion in graph-memory, offline.

    python examples/budgeted_memory.py

A fortnight of turns from one user arrives as raw text. Every few days one consolidation cycle (TIDE)
decides which turns earn a graph write within 40% of the cost of writing them all. Then the user asks
for one turn to be deleted, and recall is rebuilt without it.
"""

from graph_memory import Memory

TURNS = [
    ("2026-02-23", "I live in Manchester."),
    ("2026-02-23", "I work at Northwind Traders."),
    ("2026-03-03", "The train was late again this morning, standing room only."),
    ("2026-03-04", "I will run the Leeds half marathon every spring."),
    ("2026-03-05", "Lunch was a sandwich at my desk."),
    ("2026-03-09", "I moved to Leeds in March."),
    ("2026-03-10", "It rained all afternoon and the office was cold."),
    ("2026-03-12", "I joined Harbour Logistics in March."),
    ("2026-03-13", "Watched a film last night, it was fine."),
]
CYCLES = ["2026-03-03", "2026-03-06", "2026-03-10", "2026-03-14"]


def main() -> None:
    m = Memory(consolidation="budgeted", budget=0.4)
    spent = 0.0
    turns = iter(TURNS)
    pending = next(turns)
    for cycle in CYCLES:
        while pending and pending[0] <= cycle:
            m.add(user="u42", text=pending[1], at=pending[0])
            pending = next(turns, None)
        result = m.consolidate(at=cycle)
        spent += result.spent
        print(f"cycle {cycle}: {len(result.actions)} raw turns")
        for text, action in result.actions.items():
            print(f"  {action:<7}{text}")
    eager = sum(int(len(text.split()) * 1.3) + 1 for _, text in TURNS) / 1000
    print(f"\ngraph writes cost {spent:.3f}k tokens; writing every turn as it arrived would cost {eager:.3f}k")

    where = m.recall(user="u42", query="Where do I live and where do I work?", at="2026-03-14")
    print("\nrecall before deletion:\n" + where.context)
    m.delete(user="u42", text="I moved to Leeds in March.")
    where = m.recall(user="u42", query="Where do I live and where do I work?", at="2026-03-14")
    print(f"\nrecall after deleting the move (epoch {where.epoch}):\n" + where.context)


if __name__ == "__main__":
    main()
