# Takes a question, retrieves relevant chunks, builds a grounded prompt
# and returns the LLM's answer plus sources

from retrieve import retrieve
from llm import generate as llm_generate

RELEVANCE_THRESHOLD = 0.3 # If best chunks score below this, treat as "no answer"
REFUSAL = "I do not have enough information in the provided documents to answer that."

# Anti-hallucination mechanism (tells model to use only context, cite it & say so if unsure)
PROMPT_TEMPLATE = """You are a careful assistant answering questions about AI-governance laws and policies.

Answer the QUESTION using ONLY the CONTEXT below. Follow these rules strictly:
- Consider ALL of the provided sources, not just one. Synthesise the relevant information across every source that bears on the question.
- When sources cover different jurisdictions, instruments or perspectives, compare and combine them rather than relying on a single one.
- Cite each source you draw from by its number, like [1], [2]. Aim to use every source that is genuinely relevant.
- Base every claim on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say:
  I do not have enough information in the provided documents to answer that.
- Give a complete answer that reflects the full picture in the sources. Do not pad, but do not omit relevant material either.

CONTEXT:
{context}

QUESTION: 
{question}

ANSWER:"""

# Turn retrieved chunks into a numbered, labeled context block
def format_context(chunks):
    blocks = []
    for i, c in enumerate(chunks,1):
        source = f"{c['doc_name']} ({c['authority']})"
        blocks.append(f"[{i}] Source: {source}\n{c['text']}")
    return "\n\n".join(blocks)

def answer_question(question: str, k: int = 5):
    # Retrieve the most relevant chunks
    chunks = retrieve(question, k=k)
    
    # Relevance gate - do not call LLM at all if the best match is weak
    if not chunks or chunks[0]["score"] < RELEVANCE_THRESHOLD:
        return {"answer": REFUSAL, "sources": []}

    # Build the grounded prompt
    context = format_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Ask the LLM
    answer = llm_generate(prompt)

    # Return the answer & sources for citations
    if REFUSAL.lower()[:30] in answer.lower():
        return {"answer": answer, "sources": []}
    
    return {"answer": answer, "sources": chunks}


# Manual test
if __name__ == "__main__":
    q = "What is the capital city of Malaysia?"
    print(f"QUESTION: {q}\n" + "=" * 70)
    result = answer_question(q)
    print("\nANSWER\n" + result["answer"])

    # Only show sources if there are sources to show
    if result["sources"]:
        print("\n" + "=" * 70 + "\nSOURCES:")
        for i, s in enumerate(result["sources"], 1):
            print(f"[{i}] {s['doc_name']} ({s['authority']}) score={s['score']:.3f}")