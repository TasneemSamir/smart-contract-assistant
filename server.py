import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import config
from src.ingestion.file_parser import FileParser
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import EmbedderStore
from src.retrieval.qa_chain import QAChain
from src.guardrails.safety import GuardRails
from src.summarization.summarizer import summarizer

app = FastAPI(
    title = "Smart Contract Q&A Assistance",
    description="Upload contracts and ask questions about them",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=["*"], allow_headers=["*"]) 
file_parser = FileParser()
chunker=TextChunker()
embedder = EmbedderStore()
guardrails= GuardRails()
qa_chain:Optional[QAChain] = None
documents_cache : list =[]

class QuestionRequest(BaseModel):
    question:str

class AnswerResponse(BaseModel):
    answer:str
    sources:list
    num_sources: int
    guardrail_warnings:list=[]

## api endpoint
@app.get("/health")
async def health_check():
    #check if the server is running 
    return {"status": "healthy", "vector_store_loaded": qa_chain is not None}


@app.post('/upload')
async def upload_document(file:UploadFile = File(...)):
    ##Upload and process a document
    global qa_chain,documents_cache
    _,ext = os.path.splitext(file.filename)
    if ext.lower() not in FileParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400,detail=f"Unsupported file type: {ext}. Supported: {FileParser.SUPPORTED_EXTENSIONS}")
    
    try:
        #save file 
        file_path = os.path.join(config.UPLOAD_DIR,file.filename)
        with open(file_path,"wd") as f:
            shutil.copyfileobj(file.file,f)
        print(f"file saved:{file_path}")

        #parse text
        raw_text = file_parser.parse(file_path)
        print(f"Extracted {len(raw_text)} characters")

        #chunk text
        documents = chunker.chunk_text(raw_text,metadata={"source":file.filename})
        documents_cache = documents         #save for summarizing
        stats = chunker.get_chunk_stats(documents)
        print(f"Chunking stats: {stats}")

        #embedding and vwctor store
        vector_store = embedder.create_and_store(documents)
        #QA inirialization
        qa_chain = QAChain(vector_store)
        return {
            "message": f"Successfully processed '{file.filename}'",
            "stats": {
                "characters": len(raw_text),
                "chunks": stats["total_chunks"],
                "avg_chunk_size": stats["avg_chunk_size"],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/ask",response_model=AnswerResponse)
async def ask_question(request:QuestionRequest):
    global qa_chain

    if qa_chain is None :
            raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a document first.")
# Input guard rail check
    is_safe, message = guardrails.check_input(request.question)
    if not is_safe:
        return AnswerResponse(
            answer=message,
            sources=[],
            num_sources=0,
            guardrail_warnings=["Input blocked by guard rails"],
        )

    try:
        # Get answer from QA chain
        result = qa_chain.ask(request.question)

        # Output guard rail check
        processed_answer, metadata = guardrails.check_output(
            result["answer"],
            result["sources"],
        )

        return AnswerResponse(
            answer=processed_answer,
            sources=result["sources"],
            num_sources=result["num_sources"],
            guardrail_warnings=metadata.get("warnings", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_document():
    global documents_cache

    if not documents_cache:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet.",
        )

    try:
        summarizer = summarizer()
        summary = summarizer.summarize(documents_cache)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_session():
    global qa_chain, documents_cache
    if qa_chain:
        qa_chain.clear_history()
    qa_chain = None
    documents_cache = []
    return {"message": "Session cleared"}


# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)