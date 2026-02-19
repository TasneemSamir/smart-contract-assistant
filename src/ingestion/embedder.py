import os 
from typing import List,Optional 
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import config

class EmbedderStore:
    def __init__(self,embedding_model_name:str=None):
        model = embedding_model_name or config.EMBEDDING_MODEL
        self.embeddings=HuggingFaceEmbeddings(model_name =model,model_kwargs ={"device":"cpu"},
                                              encode_kwargs={"normalize_embeddings":True})  # normalize for cosine similarity
        
        self.vector_store :Optional[FAISS] =None


    #Embed all document chunks and create a FAISS index
    def create_and_store(self,documents:List[Document],save_path:str=None) ->FAISS:
        if not documents:
            raise ValueError('no vector embed')

        save_path = save_path or config.FAISS_INDEX_DIR
        #embedded each document page , create faiss index and store vectors with associated documents
        self.vector_store = FAISS.from_documents(documents=documents,embedding=self.embeddings) 
        #save to disk
        self.vector_store.save_local(save_path)
        print(f"FAISS index saved to {save_path}")
        print(f"Total vectors stored: {len(documents)}")
        
        return self.vector_store
    
    #load the saved FAISS index from disk
    def load_store(self,load_path:str=None)->FAISS:
        load_path = load_path or config.FAISS_INDEX_DIR
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"No FAISS index found at {load_path}. "
                "Please upload and process a document first.")
        
        self.vector_store = FAISS.load_local(load_path,embeddings=self.embeddings,allow_dangerous_deserialization=True)
        print("FAISS INDEX LOADED")

        return self.vector_store
    
    #find the most similar chuncks to a query
    def similarity_search(self,query:str,k:int=None)->List[Document]:
        if self.vector_store is None:
            raise ValueError("No vector store loaded.")
        
        k = k or config.TOP_RESULTS
        results =self.vector_store.similarity_search(query,k=k)
        return results
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = None,
    ) -> List[tuple]:
 
        if self.vector_store is None:
            raise ValueError("No vector store loaded.")

        k = k or config.TOP_K_RESULTS

        results = self.vector_store.similarity_search_with_score(query, k=k)

        return results
    

