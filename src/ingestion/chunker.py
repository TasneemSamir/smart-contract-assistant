from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List 
from config import config

class TextChunker:

    def __init__(self,chunk_size:int =None,chunk_overlap:int=None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap = self.chunk_overlap,
                                                       separators=["\n\n","\n",". "," ","" ],
                                                        length_function =len)


    #Split text into chunks, each wrapped as a LangChain Document
    def chunk_text(self,text:str,metadata:dict=None)->List[Document]:
        if not text.strip():
            return[]
        
        metadata = metadata or {}
        chunks = self.splitter.split_text(text)
        documents =[] 
        for i, chunk in enumerate(chunks):
            doc_metadata ={
                **metadata, #source file info 
                "chunk_index": i,
                "total_chunks": len(chunks),  
                "chunk_size": len(chunk),
            }
        
            documents.append(Document(page_content=chunk,metadata=doc_metadata))

        return documents
    
    def get_chunk_stats(self, documents: List[Document]) -> dict:
        if not documents:
            return {"total_chunks": 0}

        sizes = [len(d.page_content) for d in documents]
        return {
            "total_chunks": len(documents),
            "avg_chunk_size": sum(sizes) // len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes),
        }   
    
         
