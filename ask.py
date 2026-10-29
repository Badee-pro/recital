"""Ask a question over the contract corpus.

    python ask.py "How fast must a data breach be reported?"
    python ask.py "..." --method bm25 --k 3 --retrieval-only
"""

import argparse
import textwrap

from rag.retrieve import METHODS, Retriever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--method", choices=METHODS, default="dense")
    p.add_argument("--retrieval-only", action="store_true", help="skip the LLM call")
    args = p.parse_args()

    retriever = Retriever()
    chunks = retriever.search(args.question, k=args.k, method=args.method)
    print(f"\nRetrieved ({args.method}, k={args.k}):")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c.id:32s} {c.section}")

    if args.retrieval_only:
        return

    from rag.generate import answer
    ans = answer(args.question, chunks, top_similarity=retriever.top_similarity(args.question))
    print("\nAnswer:\n" + textwrap.indent(textwrap.fill(ans.text, 90), "  "))
    if ans.citations:
        print("\nSources:")
        for c in ans.citations:
            print(f"  - {c.chunk_id}: \"{textwrap.shorten(c.cited_text, 100)}\"")
    if ans.gated:
        print(f"\n(Declined without calling the model: closest clause similarity "
              f"{ans.top_similarity:.2f} is below the relevance threshold.)")
    for s in ans.uncited:
        print(f"\nWarning, no citation: \"{s}\"")


if __name__ == "__main__":
    main()
