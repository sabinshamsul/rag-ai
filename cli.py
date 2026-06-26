# Interactive CLI (Command-Line Interface)
# Loads everything once, then answer questions until user quits.

import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src")) # File import from src/ folder

from generate import answer_question

def main():
    print("=" * 70)
    print(" AI-Governance RAG (Ask Questions about AI Laws & Policies)")
    print (" Type your question and press Enter. Type 'quit' or 'exit' to leave")
    print("=" * 70)

    print("\nLoading model and index ...")
    answer_question("warm up")
    print("Ready.\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        result = answer_question(question)

        print("\n" + "-" * 70)
        print("ANSWER\n")
        print(result["answer"])

        if result["sources"]:
            print("\nSOURCES:")
            for i, s in enumerate(result["sources"], 1):
                print(f" [{i}] {s['doc_name']} ({s['authority']}) score={s['score']:.3f}")
        print("-" * 70 + "\n")

if __name__ == "__main__":
    main()