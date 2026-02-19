from typing import List 
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from config import config
from langchain_groq import ChatGroq

class Documentsummarizer:
    def __init__(self):
        self.llm =ChatGroq(model=config.GROQ_MODEL,
                           groq_api_key=config.GROQ_API_KEY,
                           temperature=0.2,         # mostyly careful
                           max_tokens=1024          #max response length
                        )
    
    def summarize(self,documents:List[Document],summary_type:str="concise")->str:
        if not documents:
            return "No documents to summarize"
        #less than 5 chunks
        if len(documents)<=5:
            return self.stuff_summarize(documents,summary_type)
        if len(documents)>=5:
            return self.map_reduce_summarize(documents,summary_type)
        
    def stuff_summarize(self,documents,summary_type):
        prompt_template = """Write a {summary_type} summary of the following document.
                         Focus on:
                            - Key parties involved
                            - Main obligations and responsibilities
                            - Important dates and deadlines
                            - Payment terms (if any)
                            - Termination conditions (if any)
                            - Any critical conditions or clauses

                            Document:
                            {text}

                            Summary:"""   
        prompt =PromptTemplate(
            template=prompt_template,
            input_variables=['text'],
            optional_variables={'summary_type':summary_type}
        )
        chain= load_summarize_chain(
            self.llm,chain_type="stuff",prompt=prompt
        )
        result = chain.invoke({'input_documents':documents})
        return result['output_text']



    def map_reduce_summarize(self, documents, summary_type):

        map_template = """Summarize this section of a document. Focus on 
                        key information, parties, obligations, and conditions:

                        Section:
                        {text}
                        Section Summary:"""
        map_prompt = PromptTemplate(
            template=map_template,
            input_variables=["text"],
        )

        reduce_template = """Combine these section summaries into one 
                            coherent {summary_type} summary. Include key parties, main obligations, 
                            important dates, and critical conditions:

                            Section Summaries:
                            {text}

                        Final Summary:"""

        reduce_prompt = PromptTemplate(
            template=reduce_template,
            input_variables=["text"],
            partial_variables={"summary_type": summary_type},
        )

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=reduce_prompt,
        )

        result = chain.invoke({"input_documents": documents})
        return result["output_text"]   
