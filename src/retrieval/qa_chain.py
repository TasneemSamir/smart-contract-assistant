from typing import List,Dict,Optional
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from config import config
from langchain_groq import ChatGroq


class QAChain:
    def __init__(self,vector_store):
        self.vector_store = vector_store
        self.llm =ChatGroq(model=config.GROQ_MODEL,
                           groq_api_key=config.GROQ_API_KEY,
                           temperature=0.2,         # mostyly careful
                           max_tokens=1024          #max response length
                        )
        self.chat_history:List=[]
        #retriever from FAISS
        self.retriever = vector_store.as_retriever(search_type ="similarity",
                                                   search_kwargs={"k": config.TOP_RESULTS})
        #QA prompt template
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

    def _get_system_prompt(self) -> str:
        return """You are a helpful contract analysis assistant. Your job is to 
        answer questions about documents based ONLY on the provided context.

        RULES:
        1. ONLY use information from the provided context to answer.
        2. If the answer is NOT in the context, say: "I cannot find this 
        information in the provided document."
        3. ALWAYS cite which section or part your answer comes from.
        4. Be precise and professional.
        5. Do NOT make up information or use external knowledge.
        6. If the question is unclear, ask for clarification.

        CONTEXT FROM DOCUMENT:
        {context}

        Answer the question based ONLY on the context above."""

    #process a user question and return an answer with the sources
    def ask(self,question:str)->Dict:
        #retrieve relevant chunks
        relevant_docs = self.retriever.invoke(question)
        context=self.format_context(relevant_docs)
        #build prompt
        prompt_messages = self.qa_prompt.format_messages(
            context=context,
            chat_history=self.chat_history,
            question=question
        )
        response = self.llm.invoke(prompt_messages)
        #extract answer text
        if hasattr(response,"content"):
            answer_text = response.content
        else:
            answer_text=str(response)

        #update conversation hist
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer_text))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
            }
            for doc in relevant_docs
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "num_sources": len(sources),
        }
    
    def format_context(self,documents:List[Document])->str:
        if not documents:
            return"No relevant information found in the document."
        context_parts=[]
        for i,doc in enumerate(documents,1):
            chunk_idx = doc.metadata.get("chunk_index","?")
            source = doc.metadata.get("source","unknown")
            context_parts.append(f"--- Source {i} (chunk {chunk_idx} from {source}) ---\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(context_parts)
    
    def clear_history(self):
        self.chat_history=[]
        print("🗑️ Conversation history cleared")
        

