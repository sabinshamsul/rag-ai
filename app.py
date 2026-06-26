
# Gradio web interface for the RAG system on Hugging Face Space.
# Reuses the same pipeline as the CLI.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr
from generate import answer_question

def respond(question):
    if not question or not question.strip():
        return "Please enter a question.", ""

    result = answer_question(question)
    answer = result["answer"]

    if result["sources"]:
        lines = []
        for i, s in enumerate(result["sources"], 1):
            lines.append(f"**[{i}]** {s['doc_name']} *({s['authority']})* — score {s['score']:.3f}")
        sources = "\n\n".join(lines)
    else:
        sources = "_No sources because the corpus did not contain enough information to answer._"

    return answer, sources


EXAMPLES = [
    "What does the law say about AI in the military?",
    "What are the penalties for violating AI regulations?",
    "How is a high-risk AI system defined?",
    "What rules exist for AI transparency and disclosure?",
]

with gr.Blocks(title="AI-Governance RAG") as demo:
    gr.Markdown(
        "# AI-Governance RAG\n"
        "Ask questions about AI laws and policies. Answers are grounded in real "
        "governance documents (the AGORA corpus) and sources cited. "
        "If the documents do not cover the question, the RAG says so rather than guessing."
    )

    question = gr.Textbox(
        label="Question?",
        placeholder="e.g. What does the law say about AI in the military?",
        lines=2,
    )
    ask_btn = gr.Button("Ask", variant="primary")

    gr.Markdown("### Answer")
    answer_box = gr.Markdown()

    gr.Markdown("### Sources")
    sources_box = gr.Markdown()

    ask_btn.click(respond, inputs=question, outputs=[answer_box, sources_box])
    question.submit(respond, inputs=question, outputs=[answer_box, sources_box])

    gr.Examples(examples=EXAMPLES, inputs=question)

if __name__ == "__main__":
    demo.launch()