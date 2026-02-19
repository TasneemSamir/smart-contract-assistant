import os 
from dotenv import load_dotenv

load_dotenv()

class config:
    #LLM 
    LLM_PROVIDER =os.getenv("LLLM_PROVIDER","groq")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    #Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",'all-MiniLM-L6-v2')
    #chunking 
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE',500))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP',50))

    #retrieval
    TOP_RESULTS = 4
    
    UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
    FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
    
    #guard rails
    RELEVANCE_THRESHOLD = 0.3

    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.UPLOAD_DIR,exist_ok = True)
        os.makedirs(cls.FAISS_INDEX_DIR,exist_ok=True)
    
    @classmethod
    def validate(cls):
        #check if the selected provider has it's api key 
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set ")
        

config.ensure_directories()