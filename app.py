import os
import shutil
import gradio as gr

from config import config
from src.ingestion.file_parser import FileParser
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import EmbedderStore
from src.retrieval.qa_chain import QAChain
from src.guardrails.safety import GuardRails
from src.summarization.summarizer import Documentsummarizer

file_parser = FileParser()
chunker = TextChunker()
embedder = EmbedderStore()
guardrails = GuardRails()

qa_chain = None
documents_cache = []

def upload_file(file):
    global qa_chain, documents_cache

    if file is None:
        return "⚠️ Please select a file to upload."

    try:
        if hasattr(file, 'name'):
            file_path = file.name
        elif isinstance(file, str):
            file_path = file
        else:
            file_path = str(file)

        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in {".pdf", ".docx"}:
            return f"❌ Unsupported file type: '{ext}'. Please upload PDF or DOCX."

        save_path = os.path.join(config.UPLOAD_DIR, filename)
        if os.path.abspath(file_path) != os.path.abspath(save_path):
            shutil.copy2(file_path, save_path)
        print(f"📁 File saved: {save_path}")

        raw_text = file_parser.parse(save_path)
        print(f"📄 Extracted {len(raw_text)} characters")

        documents = chunker.chunk_text(
            raw_text,
            metadata={"source": filename},
        )
        documents_cache = documents
        stats = chunker.get_chunk_stats(documents)
        print(f"Chunking stats: {stats}")

        vector_store = embedder.create_and_store(documents)
        qa_chain = QAChain(vector_store)

        return (
            f"✅ **Successfully processed '{filename}'**\n\n"
            f"**Document Statistics:**\n"
            f"- Characters extracted: {len(raw_text):,}\n"
            f"- Chunks created: {stats['total_chunks']}\n"
            f"- Average chunk size: {stats['avg_chunk_size']} characters\n\n"
            f"💡 Go to the **Chat** tab to ask questions!"
        )

    except Exception as e:
        print(f"❌ Full error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error processing file: {str(e)}"


def ask_question(question, chat_history):
    """
    Handle questions using simple Textbox + Markdown.
    No fancy chat components — just works!
    """
    global qa_chain

    if not question or not question.strip():
        return "", chat_history

    if qa_chain is None:
        new_entry = f"**You:** {question}\n\n**Assistant:** ⚠️ No document uploaded yet. Please go to the Upload tab first.\n\n---\n\n"
        chat_history = (chat_history or "") + new_entry
        return "", chat_history

    # Guard rail check
    is_safe, safety_msg = guardrails.check_input(question)
    if not is_safe:
        new_entry = f"**You:** {question}\n\n**Assistant:** 🛡️ {safety_msg}\n\n---\n\n"
        chat_history = (chat_history or "") + new_entry
        return "", chat_history

    try:
        result = qa_chain.ask(question)

        processed_answer, metadata = guardrails.check_output(
            result["answer"],
            result["sources"],
        )

        # Build the answer
        answer = processed_answer

        # Add sources
        if result["sources"]:
            answer += "\n\n📎 **Sources:**\n"
            for i, source in enumerate(result["sources"], 1):
                chunk_idx = source["metadata"].get("chunk_index", "?")
                source_file = source["metadata"].get("source", "unknown")
                preview = source["content"][:80].replace("\n", " ")
                answer += f"- **[{i}]** Chunk {chunk_idx} from `{source_file}`: _{preview}_\n"

        # Add warnings
        if metadata.get("warnings"):
            answer += "\n⚠️ **Warnings:**\n"
            for warning in metadata["warnings"]:
                answer += f"- {warning}\n"

        new_entry = f"**🧑 You:** {question}\n\n**🤖 Assistant:** {answer}\n\n---\n\n"
        chat_history = (chat_history or "") + new_entry
        return "", chat_history

    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        new_entry = f"**You:** {question}\n\n**Assistant:** ❌ Error: {str(e)}\n\n---\n\n"
        chat_history = (chat_history or "") + new_entry
        return "", chat_history


def summarize_document():
    global documents_cache

    if not documents_cache:
        return "⚠️ No document uploaded yet. Please upload a document first."

    try:
        doc_summarizer = Documentsummarizer()
        summary = doc_summarizer.summarize(documents_cache)
        return f"📋 **Document Summary:**\n\n{summary}"
    except Exception as e:
        print(f"❌ Summarize error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error generating summary: {str(e)}"


def clear_session():
    global qa_chain, documents_cache
    if qa_chain:
        qa_chain.clear_history()
    qa_chain = None
    documents_cache = []
    return "🗑️ Session cleared. You can upload a new document."


def clear_chat():
    """Clear chat history."""
    return ""



with gr.Blocks(
    title="Smart Contract Q&A Assistant",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 📄 Smart Contract Q&A Assistant
        Upload a contract or document and ask questions about it.

        **Steps:** 1. 📤 Upload → 2. 💬 Chat → 3. 📋 Summarize
        """
    )

    #Tab 1: Upload 
    with gr.Tab("📤 Upload Document"):
        gr.Markdown("### Upload your contract or document")

        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Select PDF or DOCX file",
                    file_types=[".pdf", ".docx"],
                )
                upload_btn = gr.Button(
                    "Process Document",
                    variant="primary",
                    size="lg",
                )
            with gr.Column(scale=3):
                upload_output = gr.Markdown(
                    value="*Upload a file to get started...*",
                )

        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Session", variant="secondary")
            clear_output = gr.Markdown()

        upload_btn.click(fn=upload_file, inputs=[file_input], outputs=[upload_output])
        clear_btn.click(fn=clear_session, inputs=[], outputs=[clear_output])

    # --- Tab 2: Chat (Simple version — just Textbox + Markdown)
    with gr.Tab("💬 Chat with Document"):
        gr.Markdown("### Ask questions about your uploaded document")

        # Chat history displayed as Markdown (simple and reliable!)
        chat_display = gr.Markdown(
            value="*Upload a document first, then ask questions here...*",
            label="Conversation",
        )

        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the termination clause?",
                scale=4,
                lines=1,
            )
            send_btn = gr.Button("Send 📨", variant="primary", scale=1)

        # Example questions
        gr.Markdown("**💡 Try these example questions:**")
        with gr.Row():
            ex1 = gr.Button("Who are the parties?", size="sm")
            ex2 = gr.Button("Payment terms?", size="sm")
            ex3 = gr.Button("Termination clause?", size="sm")
            ex4 = gr.Button("Confidentiality?", size="sm")

        clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # Connect send button and enter key
        send_btn.click(
            fn=ask_question,
            inputs=[question_input, chat_display],
            outputs=[question_input, chat_display],
        )
        question_input.submit(
            fn=ask_question,
            inputs=[question_input, chat_display],
            outputs=[question_input, chat_display],
        )

        # Connect example buttons
        ex1.click(fn=lambda h: ask_question("Who are the key parties involved in this contract?", h),
                  inputs=[chat_display], outputs=[question_input, chat_display])
        ex2.click(fn=lambda h: ask_question("What are the payment terms and conditions?", h),
                  inputs=[chat_display], outputs=[question_input, chat_display])
        ex3.click(fn=lambda h: ask_question("What is the termination or cancellation policy?", h),
                  inputs=[chat_display], outputs=[question_input, chat_display])
        ex4.click(fn=lambda h: ask_question("What are the confidentiality obligations?", h),
                  inputs=[chat_display], outputs=[question_input, chat_display])

        clear_chat_btn.click(fn=clear_chat, outputs=[chat_display])

    # --- Tab 3: Summary ---
    with gr.Tab("📋 Summary"):
        gr.Markdown("### Generate a summary of your uploaded document")
        summarize_btn = gr.Button(
            "📋 Generate Summary",
            variant="primary",
            size="lg",
        )
        summary_output = gr.Markdown(
            value="*Click the button above to generate a summary...*"
        )
        summarize_btn.click(fn=summarize_document, inputs=[], outputs=[summary_output])

    # --- Tab 4: About ---
    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ### About This Application

            **Smart Contract Q&A Assistant** is a RAG application
            built for the NVIDIA DLI course.

            #### How It Works:
            1. **UPLOAD** → PDF/DOCX parsed to text
            2. **CHUNK** → Text split into pieces
            3. **EMBED** → Chunks become vectors
            4. **STORE** → Vectors saved in FAISS
            5. **QUERY** → Question embedded & compared
            6. **RETRIEVE** → Similar chunks found
            7. **GENERATE** → LLM writes answer
            8. **CITE** → Sources attached

            #### Tech Stack:
            LangChain, FAISS, SentenceTransformers, Groq, Gradio

            ---
            *⚠️ Not a substitute for professional legal advice.*
            """
        )


# --- Launch ---
if __name__ == "__main__":
    print("🚀 Starting Smart Contract Q&A Assistant...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
    )